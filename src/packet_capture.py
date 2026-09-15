"""
src/packet_capture.py
---------------------
Real-time network packet capture that feeds the inference pipeline.

Two capture backends (tried in order):
  1. scapy   — preferred; works without Wireshark installed on most platforms.
               Requires root/Administrator on Windows.
  2. pyshark — fallback; wraps tshark (Wireshark CLI). More robust on Windows
               because tshark ships with Wireshark.

The capture runs in a background thread so Flask stays responsive.
Packets are converted to NSL-KDD-shaped feature dicts on a best-effort basis
(only the fields extractable from live traffic are populated; the rest default
to 0).  Each batch of BATCH_SIZE packets is pushed through predict_batch() and
the results land in the DB exactly like the simulate route does.

Public API
----------
    start_capture(interface, app, db, Alert, TrafficRecord)
    stop_capture()
    capture_status() -> dict
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_capture_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_status: dict = {
    "running": False,
    "interface": None,
    "backend": None,
    "packets_seen": 0,
    "alerts_generated": 0,
    "error": None,
    "started_at": None,
}
_status_lock = threading.Lock()

BATCH_SIZE = 10          # run inference every N packets
MAX_QUEUE  = 500         # cap in-memory packet queue


def capture_status() -> dict:
    with _status_lock:
        return dict(_status)


def stop_capture():
    _stop_event.set()
    if _capture_thread and _capture_thread.is_alive():
        _capture_thread.join(timeout=5)
    with _status_lock:
        _status["running"] = False


def start_capture(interface: str, flask_app, db, Alert, TrafficRecord):
    """
    Start background packet capture on *interface*.

    Args:
        interface    : NIC name (e.g. "eth0", "Wi-Fi", "\\Device\\NPF_{...}")
        flask_app    : the Flask app instance (for app_context)
        db / Alert / TrafficRecord : SQLAlchemy objects for persisting results
    """
    global _capture_thread

    if _capture_thread and _capture_thread.is_alive():
        return {"error": "Capture already running."}

    _stop_event.clear()
    with _status_lock:
        _status.update({
            "running": True,
            "interface": interface,
            "backend": None,
            "packets_seen": 0,
            "alerts_generated": 0,
            "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })

    _capture_thread = threading.Thread(
        target=_capture_loop,
        args=(interface, flask_app, db, Alert, TrafficRecord),
        daemon=True,
        name="packet-capture",
    )
    _capture_thread.start()
    return {"started": True, "interface": interface}


# ---------------------------------------------------------------------------
# Internal capture loop
# ---------------------------------------------------------------------------

def _packet_to_record(pkt) -> Optional[dict]:
    """
    Convert a scapy packet to a dict with NSL-KDD feature names.
    Only IP/TCP/UDP packets are converted; others are skipped.
    Fields not derivable from a single packet are set to 0.
    """
    try:
        # scapy layers
        from scapy.layers.inet import IP, TCP, UDP, ICMP
        if not pkt.haslayer(IP):
            return None

        ip = pkt[IP]
        proto_map = {6: "tcp", 17: "udp", 1: "icmp"}
        protocol = proto_map.get(ip.proto, "tcp")

        src_port = dst_port = 0
        flag_str  = "OTH"
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            src_port = int(tcp.sport)
            dst_port = int(tcp.dport)
            flags = tcp.flags
            if flags & 0x02:  flag_str = "S0"
            if flags & 0x12:  flag_str = "SF"
            if flags & 0x04:  flag_str = "REJ"
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            src_port = int(udp.sport)
            dst_port = int(udp.dport)
            flag_str = "SF"

        pkt_len = len(pkt)
        service = _port_to_service(dst_port)

        return {
            "duration":               0,
            "protocol_type":          protocol,
            "service":                service,
            "flag":                   flag_str,
            "src_bytes":              pkt_len,
            "dst_bytes":              0,
            "land":                   1 if ip.src == ip.dst else 0,
            "wrong_fragment":         0,
            "urgent":                 0,
            "hot":                    0,
            "num_failed_logins":      0,
            "logged_in":              1 if dst_port in (22, 80, 443, 21, 23) else 0,
            "num_compromised":        0,
            "root_shell":             0,
            "su_attempted":           0,
            "num_root":               0,
            "num_file_creations":     0,
            "num_shells":             0,
            "num_access_files":       0,
            "num_outbound_cmds":      0,
            "is_host_login":          0,
            "is_guest_login":         0,
            "count":                  1,
            "srv_count":              1,
            "serror_rate":            0.0,
            "srv_serror_rate":        0.0,
            "rerror_rate":            0.0,
            "srv_rerror_rate":        0.0,
            "same_srv_rate":          1.0,
            "diff_srv_rate":          0.0,
            "srv_diff_host_rate":     0.0,
            "dst_host_count":         1,
            "dst_host_srv_count":     1,
            "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate":   0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate":   0.0,
            "dst_host_srv_rerror_rate": 0.0,
            # metadata (not a feature)
            "label":                  "live",
            "_src_ip":                str(ip.src),
            "_dst_port":              dst_port,
        }
    except Exception:
        return None


def _port_to_service(port: int) -> str:
    _map = {
        80: "http", 443: "https", 21: "ftp", 22: "ssh",
        23: "telnet", 25: "smtp", 53: "domain", 110: "pop_3",
        143: "imap4", 3306: "sql_net", 6667: "IRC",
    }
    return _map.get(port, "other")


def _capture_loop_scapy(interface, packet_queue, stop_event):
    """Scapy capture — runs in its own thread, pushes to packet_queue."""
    from scapy.all import sniff

    def handle(pkt):
        if stop_event.is_set():
            return
        rec = _packet_to_record(pkt)
        if rec and len(packet_queue) < MAX_QUEUE:
            packet_queue.append(rec)

    sniff(iface=interface, prn=handle, store=False,
          stop_filter=lambda _: stop_event.is_set())


def _capture_loop_pyshark(interface, packet_queue, stop_event):
    """pyshark fallback — slower but works without root on Windows + Wireshark."""
    import pyshark

    cap = pyshark.LiveCapture(interface=interface, display_filter="ip")
    for pkt in cap.sniff_continuously():
        if stop_event.is_set():
            cap.close()
            break
        try:
            proto = getattr(pkt, "transport_layer", "tcp") or "tcp"
            src_bytes = int(pkt.length) if hasattr(pkt, "length") else 0
            dst_port = 0
            flag_str = "OTH"
            if hasattr(pkt, "tcp"):
                dst_port = int(pkt.tcp.dstport)
                flags = int(pkt.tcp.flags, 16) if pkt.tcp.flags else 0
                flag_str = "SF" if flags & 0x12 else ("S0" if flags & 0x02 else "OTH")
            elif hasattr(pkt, "udp"):
                dst_port = int(pkt.udp.dstport)
                flag_str = "SF"
            rec = {k: 0 for k in [
                "duration","src_bytes","dst_bytes","land","wrong_fragment",
                "urgent","hot","num_failed_logins","logged_in","num_compromised",
                "root_shell","su_attempted","num_root","num_file_creations",
                "num_shells","num_access_files","num_outbound_cmds","is_host_login",
                "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
                "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
                "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
                "dst_host_same_srv_rate","dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
                "dst_host_serror_rate","dst_host_srv_serror_rate",
                "dst_host_rerror_rate","dst_host_srv_rerror_rate",
            ]}
            rec.update({
                "protocol_type": proto.lower(),
                "service": _port_to_service(dst_port),
                "flag": flag_str,
                "src_bytes": src_bytes,
                "same_srv_rate": 1.0,
                "dst_host_same_srv_rate": 1.0,
                "label": "live",
                "_dst_port": dst_port,
            })
            if len(packet_queue) < MAX_QUEUE:
                packet_queue.append(rec)
        except Exception:
            pass


def _capture_loop(interface, flask_app, db, Alert, TrafficRecord):
    """Main capture loop: choose backend, batch packets, run inference."""
    from src.predict import predict_batch

    packet_queue: deque = deque()

    # Choose backend
    backend = None
    sniffer_thread = None
    try:
        import scapy.all  # noqa — just probe availability
        backend = "scapy"
        sniffer_thread = threading.Thread(
            target=_capture_loop_scapy,
            args=(interface, packet_queue, _stop_event),
            daemon=True,
        )
    except ImportError:
        pass

    if backend is None:
        try:
            import pyshark  # noqa
            backend = "pyshark"
            sniffer_thread = threading.Thread(
                target=_capture_loop_pyshark,
                args=(interface, packet_queue, _stop_event),
                daemon=True,
            )
        except ImportError:
            pass

    if backend is None:
        with _status_lock:
            _status["error"] = "Neither scapy nor pyshark is installed. Run: pip install scapy"
            _status["running"] = False
        return

    with _status_lock:
        _status["backend"] = backend

    log.info("Packet capture started on %s using %s", interface, backend)
    sniffer_thread.start()

    batch: list = []
    while not _stop_event.is_set():
        # Drain the queue into batch
        while packet_queue and len(batch) < BATCH_SIZE:
            batch.append(packet_queue.popleft())
            with _status_lock:
                _status["packets_seen"] += 1

        if len(batch) >= BATCH_SIZE:
            try:
                results = predict_batch(batch)
                with flask_app.app_context():
                    new_alerts = 0
                    for res in results:
                        tr = TrafficRecord(
                            timestamp=res["timestamp"],
                            attack_category=res["attack_category"],
                            risk_score=res["risk_score"],
                            is_alert=res["is_alert"],
                        )
                        db.session.add(tr)
                        if res["is_alert"]:
                            al = Alert(
                                timestamp=res["timestamp"],
                                attack_label=res["attack_label"],
                                attack_category=res["attack_category"],
                                risk_score=res["risk_score"],
                                clf_confidence=res["clf_confidence"],
                                anomaly_score=res["anomaly_score"],
                                binary_pred=res["binary_pred"],
                                reason_string=res["reason_string"],
                                explain_method=res["explain_method"],
                                top_features_json=json.dumps(res["top_features"]),
                                technique_id=res["technique_id"],
                                technique_name=res["technique_name"],
                                tactic=res["tactic"],
                                technique_reasoning=res["technique_reasoning"],
                                nearest_label=res["nearest_label"],
                                nearest_similarity=res["nearest_similarity"],
                                incident_summary=res["incident_summary"],
                                summary_mode=res["summary_mode"],
                            )
                            db.session.add(al)
                            new_alerts += 1
                    db.session.commit()
                with _status_lock:
                    _status["alerts_generated"] += new_alerts
                batch.clear()
            except Exception as exc:
                log.warning("Capture inference batch failed: %s", exc)
                batch.clear()
        else:
            time.sleep(0.1)

    sniffer_thread.join(timeout=3)
    log.info("Packet capture stopped.")
