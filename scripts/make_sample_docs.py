"""Generate the sample PDF corpus used for the demo and the evaluation set.

The six PDFs are short educational summaries WRITTEN FOR THIS REPO (they are not copies of any
standard or textbook). Each section is forced onto its own page, so evaluation page numbers are
deterministic. Run:  python scripts/make_sample_docs.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_docs"

DOCS = {
    "can_bus_fundamentals": {
        "title": "CAN Bus Fundamentals",
        "pages": [
            ("Overview and physical layer", [
                "The Controller Area Network (CAN) protocol was developed by Bosch and introduced in 1986. It was created so that "
                "microcontrollers and devices inside a vehicle could communicate without a host computer. CAN is standardised as ISO 11898.",
                "CAN is a multi-master, message-based broadcast bus: every node sees every message, and any node may start transmitting when the "
                "bus is idle. Messages are identified by content rather than by the address of a destination node.",
                "On the physical layer CAN uses differential signalling on a twisted pair of wires called CAN_H and CAN_L. The difference between "
                "the two lines carries the bit value, which makes the bus robust against electromagnetic noise. A 120 ohm termination resistor is "
                "placed at each end of the bus to prevent signal reflections. Without correct termination, reflections corrupt the waveform and "
                "cause errors at higher bit rates.",
                "The bus has two logical states. The dominant state represents a logical 0 and the recessive state represents a logical 1. "
                "When one node drives the bus dominant while another drives it recessive, the bus ends up dominant.",
            ]),
            ("Frame format and arbitration", [
                "A standard CAN frame has an 11-bit identifier. The extended frame format has a 29-bit identifier. A classical CAN data field "
                "carries between 0 and 8 bytes. Each frame also contains a 15-bit CRC field for error detection and an acknowledge slot in which "
                "receivers confirm correct reception.",
                "When two nodes begin transmitting at the same moment, CAN resolves the conflict through non-destructive bitwise arbitration. "
                "Each transmitter monitors the bus while sending its identifier. If a node sends a recessive bit but reads a dominant bit, it has lost "
                "arbitration, stops transmitting and becomes a receiver. The message with the lower numerical identifier wins because it contains "
                "dominant bits earlier, so a lower identifier means a higher priority. The winning message is not damaged and no time is lost.",
                "To keep receivers synchronised, CAN uses bit stuffing: after five consecutive bits of the same polarity the transmitter inserts a "
                "bit of the opposite polarity, which the receivers remove again.",
            ]),
            ("Speed, CAN FD and error handling", [
                "Classical CAN supports bit rates up to 1 Mbit/s. The achievable bus length falls as the bit rate rises: roughly 40 m at 1 Mbit/s, "
                "about 100 m at 500 kbit/s and up to about 500 m at 125 kbit/s.",
                "CAN FD (flexible data-rate) extends the classical protocol. A CAN FD frame can carry up to 64 data bytes instead of 8, and the data "
                "phase can run at a higher bit rate than the arbitration phase when the bit rate switch (BRS) bit is set. Data-phase rates of 5 to "
                "8 Mbit/s are commonly quoted.",
                "CAN has five error types: bit error, stuff error, CRC error, form error and acknowledgement error. Every node keeps a transmit error "
                "counter (TEC) and a receive error counter (REC). A node is error-active while both counters are low. When either counter exceeds 127 "
                "the node becomes error-passive, and when the transmit error counter exceeds 255 the node goes bus-off and stops participating "
                "until it is reset. This mechanism isolates a faulty node so that it cannot block the whole network.",
            ]),
        ],
    },
    "j1939_primer": {
        "title": "SAE J1939 Primer",
        "pages": [
            ("What J1939 is", [
                "SAE J1939 is a family of standards from the Society of Automotive Engineers for communication over CAN in heavy-duty vehicles, "
                "diesel engines, generator sets and agricultural and construction machinery. It builds a complete application layer on top of "
                "the CAN data link so that equipment from different manufacturers can exchange engine and drivetrain data.",
                "J1939 networks use 29-bit extended CAN identifiers. The most common physical layer runs at 250 kbit/s on a shielded twisted pair, "
                "while a newer variant uses 500 kbit/s. Most engine data is broadcast periodically, and a request mechanism lets a device ask for "
                "a message that is not sent cyclically.",
            ]),
            ("Identifier structure and addressing", [
                "The 29-bit J1939 identifier is split into fields: a 3-bit priority, a reserved bit, a data page bit, an 8-bit PDU Format (PF), "
                "an 8-bit PDU Specific (PS) field and an 8-bit source address (SA). The reserved bit, data page bit, PF and PS together form the "
                "18-bit Parameter Group Number (PGN), which identifies what the message contains.",
                "If the PF value is below 240 the message is of type PDU1 and the PS field holds the destination address, so the message is sent "
                "to one node. If PF is 240 or higher the message is PDU2, a broadcast message, and the PS field acts as a group extension. "
                "Source addresses are 8 bits wide: 254 is the null address and 255 is the global address.",
                "Control messages usually default to priority 3, while informational and most other messages default to priority 6. A lower "
                "priority number wins arbitration. Devices obtain their source address with the address claiming procedure, which uses PGN 60928.",
            ]),
            ("Common engine messages and signal scaling", [
                "Each value in a message is a Suspect Parameter Number (SPN). To convert a raw value, multiply by the resolution and add the "
                "offset. Several widely used messages are listed here.",
                "Electronic Engine Controller 1 (EEC1) is PGN 61444 (0xF004). It carries engine speed as SPN 190 in bytes 4 and 5, with a "
                "resolution of 0.125 rpm per bit.",
                "Engine Temperature 1 (ET1) is PGN 65262 (0xFEEE). Engine coolant temperature is SPN 110 in byte 1. It has a resolution of "
                "1 degree C per bit and an offset of -40 degrees C, so a raw value of 130 means 90 degrees C.",
                "Fuel Economy (LFE) is PGN 65266 (0xFEF2). The engine fuel rate is SPN 183 in bytes 1 and 2 with a resolution of 0.05 litres per "
                "hour per bit. Cruise Control and Vehicle Speed (CCVS) is PGN 65265 (0xFEF1) and carries wheel-based vehicle speed as SPN 84 in "
                "bytes 2 and 3 with a resolution of 1/256 km/h per bit.",
            ]),
            ("Diagnostics and long messages", [
                "J1939 diagnostics use diagnostic messages (DMs). DM1 is PGN 65226 (0xFECA) and reports the currently active diagnostic trouble "
                "codes (DTCs). By default an ECU broadcasts DM1 once per second while a fault is active, and also sends it immediately when the "
                "fault status changes. DM2 (PGN 65227) lists previously active DTCs, and DM11 (PGN 65235) is used to clear active DTCs.",
                "Each DTC contains a 19-bit SPN that names the faulty signal, a 5-bit Failure Mode Identifier (FMI) that describes the type of "
                "fault, a 7-bit occurrence count and a conversion method bit. DM1 also carries the lamp status: malfunction indicator lamp, red "
                "stop lamp, amber warning lamp and protect lamp.",
                "A CAN frame holds only 8 data bytes, so J1939 defines a transport protocol for longer messages. A sender announces the transfer "
                "with the connection management message TP.CM (PGN 60416) and then sends the data in TP.DT frames (PGN 60160). The transport "
                "protocol can move up to 1785 bytes in one message.",
            ]),
        ],
    },
    "predictive_maintenance_primer": {
        "title": "Predictive Maintenance Primer",
        "pages": [
            ("Maintenance strategies", [
                "There are three broad maintenance strategies. Reactive maintenance, also called run-to-failure, repairs equipment only after "
                "it breaks. It has no planning cost but can cause expensive unplanned downtime and secondary damage.",
                "Preventive maintenance services equipment at fixed time or usage intervals regardless of its actual state. It reduces surprise "
                "failures but often replaces parts that still have useful life left.",
                "Predictive maintenance, also known as condition-based maintenance, uses sensor data to estimate the actual health of the "
                "equipment and schedules work only when it is needed. The aim is to cut unplanned downtime while avoiding unnecessary "
                "replacements. The difference from preventive maintenance is that the trigger is the measured condition, not the calendar.",
            ]),
            ("Remaining useful life and the C-MAPSS data", [
                "Remaining Useful Life (RUL) is the time left before an asset reaches a failure threshold, usually measured in operating "
                "cycles. Approaches include physics-based models, data-driven models learned from run-to-failure histories, and hybrids.",
                "A popular benchmark is NASA's C-MAPSS turbofan engine degradation simulation. It has four subsets and 21 sensor channels. "
                "FD001 has one operating condition and one fault mode (high-pressure compressor degradation) with 100 training engines. FD002 has "
                "six operating conditions and one fault mode. FD003 has one operating condition and two fault modes (high-pressure compressor and fan "
                "degradation). FD004 has six operating conditions and two fault modes and is the hardest subset.",
                "In the training files every engine is run until it fails. In the test files the series stops before failure and a separate file "
                "gives the true RUL at the last observed cycle.",
            ]),
            ("Evaluating RUL models", [
                "The most common metric is the root mean squared error (RMSE) of predicted versus true RUL. Because predicting failure too late "
                "is more harmful than predicting it too early, the PHM08 challenge introduced an asymmetric scoring function. With d equal to "
                "predicted minus actual RUL, the score is exp(-d/13) - 1 when d is negative and exp(d/10) - 1 otherwise, so late predictions are "
                "penalised more heavily than early ones.",
                "Many studies clip the training RUL target at 125 cycles, giving a piecewise linear target. In early life the sensor readings of "
                "a healthy engine look practically identical, so forcing a model to separate 200 cycles from 150 cycles left only adds noise.",
                "A predicted RUL is only useful when it triggers an action. An alert threshold turns it into a decision, and choosing it is a "
                "trade-off: a high threshold catches more real failures but raises more false alarms, while a low threshold misses failures. "
                "The threshold should be set from the relative cost of a missed failure and an unnecessary inspection.",
            ]),
        ],
    },
    "diesel_engine_diagnostics": {
        "title": "Diesel Engine Diagnostics Basics",
        "pages": [
            ("How a diesel engine works", [
                "A diesel engine is a compression-ignition engine. In the four-stroke cycle the piston performs intake, compression, power and "
                "exhaust strokes. During the intake stroke only air enters the cylinder.",
                "The air is compressed so strongly, with compression ratios typically between 14:1 and 25:1, that its temperature rises above the "
                "autoignition temperature of diesel fuel. Fuel injected near the end of compression ignites by itself, so a diesel engine has no "
                "spark plugs. The high compression ratio is the reason diesels are more thermally efficient than petrol engines.",
            ]),
            ("Symptoms and what they indicate", [
                "Condition monitoring of a diesel engine relies on a few key measurements: coolant temperature, oil pressure, turbocharger boost "
                "pressure, exhaust gas temperature (EGT), crankcase blow-by and vibration.",
                "A high exhaust gas temperature suggests overfuelling or restricted airflow. Low oil pressure at idle points to worn bearings or "
                "a failing oil pump. Excessive blow-by indicates worn piston rings or cylinder liners.",
                "The colour of the exhaust smoke is a quick diagnostic clue. Black smoke means incomplete combustion from too much fuel or too "
                "little air. White smoke means unburnt fuel or coolant entering the combustion chamber. Blue smoke means that lube oil is being "
                "burned, often because of worn valve seals or piston rings.",
            ]),
            ("Exhaust aftertreatment", [
                "Modern diesel engines use several systems to reduce emissions. Exhaust gas recirculation (EGR) feeds part of the exhaust back "
                "into the intake, which lowers peak combustion temperature and therefore the formation of nitrogen oxides (NOx).",
                "A diesel particulate filter (DPF) traps soot from the exhaust. As soot accumulates the filter must be regenerated by burning "
                "the soot off at high temperature; regeneration can be passive, active or forced.",
                "Selective catalytic reduction (SCR) converts NOx into harmless nitrogen and water. It works by injecting diesel exhaust fluid "
                "(DEF), a solution of 32.5 percent urea in deionised water, into the exhaust stream upstream of a catalyst. If the DEF tank runs "
                "empty, the engine control unit will usually derate the engine.",
            ]),
        ],
    },
    "fhss_wireless_primer": {
        "title": "Frequency Hopping and BER Primer",
        "pages": [
            ("Frequency-hopping spread spectrum", [
                "In frequency-hopping spread spectrum (FHSS) the transmitter rapidly switches its carrier among many frequency channels following "
                "a pseudo-random sequence that the receiver also knows. The time spent on one frequency is the dwell time, and the number of "
                "changes per second is the hop rate.",
                "FHSS resists narrowband interference, because only the hops that land on the interfered channel are damaged. It also makes "
                "eavesdropping harder and averages out frequency-selective fading. Slow hopping sends several symbols per hop, while fast "
                "hopping uses several hops per symbol.",
                "Classic Bluetooth is a well-known FHSS system. It hops over 79 channels of 1 MHz in the 2.4 GHz ISM band at 1600 hops per second.",
            ]),
            ("Modulation and bit error rate", [
                "For coherent BPSK over an additive white Gaussian noise (AWGN) channel the bit error rate is Q(sqrt(2 Eb/N0)), which equals "
                "0.5 times erfc(sqrt(Eb/N0)). Here Eb is the energy per bit and N0 is the noise power spectral density.",
                "QPSK with Gray coding has the same bit error rate as BPSK for a given Eb/N0, while sending two bits per symbol and so using half the "
                "bandwidth. BPSK needs an Eb/N0 of about 9.6 dB to reach a bit error rate of 1e-5, and about 6.8 dB for 1e-3.",
                "Simulating a link means generating random bits, mapping them to symbols, adding Gaussian noise scaled to the target Eb/N0, "
                "demodulating and counting errors. To estimate a very low error rate with reasonable confidence, enough error events must be "
                "observed, typically at least a hundred.",
            ]),
            ("Synchronisation and partial-band interference", [
                "A frequency-hopping receiver must align both the timing and the hop sequence with the transmitter. Acquisition searches for the "
                "correct hop phase, and tracking then keeps the receiver locked as clocks drift.",
                "In partial-band interference an interferer covers only a fraction of the hopping band. For a worst-case fraction, the bit error "
                "rate of uncoded BPSK falls only inversely with the signal-to-interference ratio, instead of exponentially as in plain AWGN. This "
                "is why FHSS systems rely on error-correcting codes, interleaving and diversity to survive intentional or accidental jamming.",
            ]),
        ],
    },
    "network_intrusion_primer": {
        "title": "Network Intrusion Detection Primer",
        "pages": [
            ("Types of intrusion detection", [
                "An intrusion detection system (IDS) monitors activity and raises alerts about malicious behaviour. A network-based IDS (NIDS) "
                "inspects traffic on a network segment, while a host-based IDS (HIDS) watches logs and system calls on one machine.",
                "Signature-based detection matches traffic against patterns of known attacks. It is precise on known threats but blind to new "
                "ones. Anomaly-based detection learns what normal behaviour looks like and flags deviations, so it can find novel attacks but "
                "tends to produce more false positives.",
            ]),
            ("The NSL-KDD dataset", [
                "NSL-KDD is an improved version of the KDD Cup 99 intrusion dataset. It removes redundant records from the training set and "
                "duplicates from the test set so that classifiers are not biased toward frequent records. The training file KDDTrain+ contains "
                "125,973 records and the test file KDDTest+ contains 22,544 records.",
                "Each connection record has 41 features plus a label. Attacks fall into four families: denial of service (DoS), probing, remote "
                "to local (R2L) and user to root (U2R). The test set contains attack types that do not appear in the training set, which makes "
                "it a realistic test of generalisation.",
            ]),
            ("Evaluating detectors", [
                "Accuracy can be misleading for intrusion detection because of class imbalance: a classifier that labels every record as normal "
                "can still score a high accuracy when attacks are rare. Precision, recall and F1 are more informative.",
                "The false positive rate matters in practice, since too many false alarms cause alert fatigue and real attacks get ignored. "
                "U2R is the rarest family and the hardest to detect, and a model that looks excellent on validation data can lose a lot of "
                "performance on the separate test set because of distribution shift.",
            ]),
        ],
    },
}


def build(name: str, spec: dict) -> Path:
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=11, leading=15, spaceAfter=8)
    path = OUT / f"{name}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2.2 * cm, bottomMargin=2.2 * cm, title=spec["title"], author="docqa-rag sample corpus")
    story = []
    for i, (heading, paras) in enumerate(spec["pages"]):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(f"{spec['title']}: {heading}.", h1))
        story.append(Spacer(1, 4))
        for p in paras:
            story.append(Paragraph(p, body))
    doc.build(story)
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    from pypdf import PdfReader

    for name, spec in DOCS.items():
        p = build(name, spec)
        n = len(PdfReader(str(p)).pages)
        assert n == len(spec["pages"]), f"{name}: {n} pages, expected {len(spec['pages'])} (a section overflowed)"
        print(f"wrote {p.name}  ({n} pages)")
