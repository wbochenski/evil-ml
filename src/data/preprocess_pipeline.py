import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

INPUT_PATH = "data/processed/data.csv"
OUTPUT_PATH = "data/processed/processed_data.csv"

COLS_TO_DROP = [
    "frame.number", "frame.time", "frame.time_epoch", "frame.time_relative", 
    "frame.time_delta", "frame.time_delta_displayed", "tcp.payload", "udp.payload", 
    "data.data", "http.file_data", "ssh.padding_string", "wlan_rsna_eapol.keydes.data",
    "wlan.sa", "wlan.da", "wlan.bssid", "wlan.ra", "wlan.ta", "ip.src", "ip.dst",
    "radiotap.mactime", "radiotap.timestamp.ts", "wlan.fixed.timestamp", 
    "wlan_radio.timestamp", "ssh.protocol", "dns.flags.opcode", "wlan.ssid",
    "radiotap.present.tsft", "nbns", "tcp.checksum", "tcp.seq_raw",
    "wlan.rsn.ie.igtk.key", "arp.src.hw_mac", "arp.dst.hw_mac", "arp.proto.type", 
    "ip.version", "arp", "nbss.continuation_data", "wlan.analysis.kck", 
    "wlan.analysis.kek", "wlan.rsn.ie.gtk.key", "wlan_rsna_eapol.keydes.nonce",
    "wlan.country_info.fnm", "wlan.country_info.code", "wlan.tag", "wlan.tag.length",
    "dns.a", "dns.qry.name", "dns.resp.name", "http.host", "http.request.uri.path", 
    "http.request.uri.query", "http.request.uri.query.parameter", "http.response_for.uri", 
    "http.referer", "tls.handshake.session_ticket_length", "http.request.line", 
    "http.response.line", "json.value.string", "json.key", "http.request.full_uri", "http.location",
    "wlan_radio.signal_dbm", "radiotap.dbm_antsignal", "llc", "dhcp.hw.mac_addr" 
]

CATEGORICAL_COLS = [
    'tcp.analysis', 'tcp.analysis.flags', 'tcp.analysis.retransmission',
    'tcp.checksum.status', 'nbss.type', 'smb2.buffer_code', 'smb2.fid', 'smb2.pid', 
    'smb2.protocol_id', 'smb2.sesid', 'smb2.tid', 'dhcp', 'dhcp.cookie',  
    'dhcp.id', 'dhcp.ip.client', 'dhcp.ip.relay', 'dhcp.ip.server', 'dhcp.option.broadcast_address',
    'dhcp.option.dhcp_server_id', 'dhcp.option.router', 'mdns', 'dns', 'dns.ptr.domain_name', 
    'ssdp', 'http.connection', 'http.last_modified', 'http.request.method', 'http.request.version', 
    'http.response.code.desc', 'http.response.phrase', 'http.response.version', 'tls.app_data_proto',
    'tls.handshake.version', 'http.content_type', 'http.server'
]

MULTI_COLS = ["dns.resp.ttl", "tls.handshake.extension.type", "tls.handshake.extensions_key_share_group", "tcp.option_len"]
HEX_COLS = ["dns.id", "tls.record.version", "tcp.checksum", "wlan.fc.ds"]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def is_multicast(mac):
    """
    Determines if a MAC address is multicast/broadcast based on the Least Significant 
    Bit (LSB) of the first octet.
    Returns: 1 (Multicast), 0 (Unicast), or np.nan for invalid/missing data.
    """
    if pd.isna(mac) or mac == "Unknown": return np.nan
    
    return int(str(mac).split(":")[0], 16) & 1

def parse_hex(x):
    """
    Attempts to extract and convert the first segment of a hyphen-separated 
    hexadecimal string into a base-10 integer.
    """
    try: 
        return int(str(x).split('-')[0], 16)
    except (ValueError, TypeError, IndexError): 
        return np.nan

def extract_stats(x):
    """
    Parses a string of hyphen-separated numeric values to calculate basic statistics.
    Returns: A Pandas Series containing [Mean, Max, Count].
    """
    try:
        vals = [float(v) for v in str(x).split("-") if v.replace('.','').isdigit()]
        if vals:
            return pd.Series([np.mean(vals), np.max(vals), len(vals)])
        
        return pd.Series([np.nan, np.nan, np.nan])
    except Exception: 
        return pd.Series([np.nan, np.nan, np.nan])

# ==========================================
# MAIN PIPELINE
# ==========================================

def main():
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    
    # dictionary to collect all new columns to avoid fragmentation
    new_cols = {}

    # 1. Basic Label Encoding (Target)
    if "Label" in df.columns:
        df["Label"] = df["Label"].map({"Evil_Twin": 1, "Normal": 0})

    # 2. MAC & Behavioral Specific Features
    new_cols["is_broadcast"] = (df["wlan.da"] == "ff:ff:ff:ff:ff:ff").astype(float)
    new_cols["is_multicast"] = df["wlan.da"].apply(is_multicast)
    new_cols["packet_rate"] = (1 / pd.to_numeric(df["frame.time_delta"], errors='coerce')).replace([np.inf, -np.inf], 0)
    new_cols["wlan_tag_count"] = df["wlan.tag"].apply(lambda x: len(str(x).split("-")) if pd.notna(x) else 0)
    new_cols["dns_qry_is_local"] = df["dns.qry.name"].str.contains("local", case=False, na=False).astype(float)

    # 3. Extracted Stats (Multi-columns)
    for col in MULTI_COLS:
        if col in df.columns:
            stats_df = df[col].apply(extract_stats)
            new_cols[f"{col}_mean"] = stats_df[0]
            new_cols[f"{col}_max"] = stats_df[1]
            new_cols[f"{col}_count"] = stats_df[2]

    # 4. Hex Conversions
    for col in HEX_COLS:
        if col in df.columns:
            new_cols[col + '_int'] = df[col].apply(parse_hex)

    # 5. Categorical Encoding
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            cleaned_col = df[col].apply(lambda x: str(x).split('-')[0].lower() if pd.notna(x) else "unknown")
            new_cols[col + '_code'] = LabelEncoder().fit_transform(cleaned_col)

    # 6. Prevents PerformanceWarning
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    # 7. Drop Raw/Leaky/Invariant Columns
    invariant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    drop_targets = COLS_TO_DROP + CATEGORICAL_COLS + MULTI_COLS + invariant_cols
    df.drop(columns=[c for c in drop_targets if c in df.columns], inplace=True, errors="ignore")

    # 8. Type Coercion & Imputation
    for col in df.columns:
        if col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median())).fillna(0)

    # 9. Normalization
    cols_to_normalize = [c for c in df.columns if c != 'Label' and df[c].nunique() > 2]
    scaler = MinMaxScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    # 10. Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Processing complete. Data saved to {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    main()