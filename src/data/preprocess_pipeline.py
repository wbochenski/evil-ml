import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
# NOTE: artifacts contains the fitted encoders, scalers, and master schema to ensure consistency between training and inference 
ARTIFACT_DIR = "data/artifacts"

# NETWORK SPECIFIC IDENTIFIERS:
# Were dropped because they betray the Evil Twin by identifying the user's device or access point:
# ip.src, ip.dst, wlan.sa, wlan.da, wlan.bssid, wlan.ra, wlan.ta, arp.src.hw_mac, arp.dst.hw_mac, dhcp.hw.mac_addr

# TIMESTAMP COLUMNS:
# frame.number: It is just the row index in the capture file, which would teach the model to memorize arbitrary row numbers instead of attack behaviors.
# frame.time: It is a human-readable string (like "Mar 18") that ML models cannot calculate, and calendar dates don't trigger attacks.
# frame.time_epoch: It is an ever-increasing absolute timestamp that would cause the model to overfit to the specific date and time the lab data was recorded.
# frame.time_relative: It only measures the time elapsed since you clicked "start" on the recording, which is a lab artifact, not a network behavior.
# radiotap.mactime: It tracks the capturing network adapter's total uptime, which has no relationship to the maliciousness of the packets it receives.
# radiotap.timestamp.ts: It measures how long the Access Point has been running (TSF timer), which does not help identify an Evil Twin attack.
# wlan.fixed.timestamp: It is another Access Point uptime counter found in management frames that adds numerical noise without offering predictive value.
# wlan_radio.timestamp: It is a local hardware capture timer that ties the data to your specific listening device rather than the actual network traffic.

# RAW PAYLOADS & ENCRYPTED FIELDS:
# tcp.payload: It contains raw, unstructured application data that tabular ML models cannot process mathematically without deep learning or NLP.
# udp.payload: It is highly variable, raw data that adds massive noise and high cardinality without providing consistent network behavior features.
# data.data: It represents arbitrary, unparsed packet bytes which act as random noise unless explicitly engineered into specific features.
# http.file_data: It contains the actual contents of transferred files or webpages, which is irrelevant to detecting the network-layer behavior of an Evil Twin.
# ssh.padding_string: It consists of randomized padding bytes designed specifically to obscure packet lengths, offering absolutely zero learnable patterns.
# wlan_rsna_eapol.keydes.data: It holds encrypted key information from WPA handshakes, which appears as pure, unpredictable randomness to an ML model.
# wlan.rsn.ie.igtk.key: It is a pseudo-random cryptographic key used for management frames, providing no predictable pattern to identify malicious intent.
# wlan.rsn.ie.gtk.key: It is a pseudo-random group encryption key that changes frequently and cannot be used to generalize attack signatures.
# wlan_rsna_eapol.keydes.nonce: It is a randomly generated, single-use cryptographic number (number used once) that inherently prevents any pattern recognition by design.

# HIGH CARDINALITY COLUMNS:
# dns.qry.name, dns.resp.name, http.host, http.request.uri.path, http.request.uri.query, http.referer, http.request.line, json.value.string, wlan.ssid

# THE REST OF THE COLUMNS WERE DROPPED BECAUSE THEY WERE INVARIANT (ONLY ONE VALUE)

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

invariant_cols = ['frame.encap_type', 'radiotap.channel.flags.cck', 'radiotap.channel.flags.ofdm', 'radiotap.channel.freq', 'radiotap.rxflags', 'radiotap.vendor_oui', 'wlan.fcs.bad_checksum', 'wlan.fixed.beacon', 'wlan.fixed.capabilities.ess', 'wlan.fixed.capabilities.ibss', 'wlan.rsn.ie.gtk.key', 'wlan.rsn.ie.igtk.key', 'wlan.rsn.ie.pmkid', 'wlan.ssid', 'wlan_radio.channel', 'wlan_radio.frequency', 'wlan_rsna_eapol.keydes.msgnr', 'wlan_rsna_eapol.keydes.data_len', 'wlan_rsna_eapol.keydes.key_info.key_mic', 'eapol.keydes.key_len', 'arp.hw.type', 'arp.proto.type', 'arp.hw.size', 'arp.proto.size', 'icmpv6.ni.nonce', 'tcp.analysis.reused_ports', 'nbns', 'nbss.type', 'ldap', 'smb.access.generic_execute', 'smb.access.generic_read', 'smb.access.generic_write', 'smb.flags.notify', 'smb.flags.response', 'smb.flags2.nt_error', 'smb.flags2.sec_sig', 'smb.mid', 'smb.nt_status', 'smb.server_component', 'smb.pid.high', 'smb.tid', 'smb2.acct', 'smb2.auth_frame', 'smb2.data_offset', 'smb2.domain', 'smb2.fid', 'smb2.filename', 'smb2.header_len', 'smb2.host', 'smb2.pid', 'smb2.previous_sesid', 'smb2.protocol_id', 'smb2.session_flags', 'smb2.write_length', 'dhcp.client_id.duid_ll_hw_type', 'dhcp.cookie', 'dhcp.hw.addr_padding', 'dhcp.ip.client', 'dhcp.ip.relay', 'dhcp.option.broadcast_address', 'dhcp.option.dhcp_server_id', 'dhcp.option.router', 'dhcp.option.vendor.bsdp.message_type', 'dns.ptr.domain_name', 'dns.resp.len', 'dns.retransmit_response', 'ssdp', 'http.next_request_in', 'http.next_response_in', 'http.response.version', 'http.response_in', 'ssh.cookie', 'ssh.compression_algorithms_client_to_server_length', 'ssh.compression_algorithms_server_to_client_length', 'ssh.direction', 'ssh.dh_gex.max', 'ssh.dh_gex.min', 'ssh.dh_gex.nbits', 'ssh.encryption_algorithms_client_to_server_length', 'ssh.encryption_algorithms_server_to_client_length', 'ssh.host_key.length', 'ssh.host_key.type_length', 'ssh.kex_algorithms_length', 'ssh.mac_algorithms_client_to_server_length', 'ssh.mac_algorithms_server_to_client_length', 'ssh.message_code', 'ssh.mpint_length', 'ssh.packet_length', 'ssh.packet_length_encrypted', 'ssh.padding_length', 'ssh.padding_string', 'ssh.protocol', 'ssh.server_host_key_algorithms_length', 'tls.connection_id']

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def is_multicast(mac):
    if pd.isna(mac) or mac == "Unknown": return np.nan
    return int(str(mac).split(":")[0], 16) & 1

def parse_hex(x):
    try: return int(str(x).split('-')[0], 16)
    except (ValueError, TypeError, IndexError): return np.nan

def extract_stats(x):
    try:
        vals = [float(v) for v in str(x).split("-") if v.replace('.','').isdigit()]
        if vals: return pd.Series([np.mean(vals), np.max(vals), len(vals)])
        return pd.Series([np.nan, np.nan, np.nan])
    except Exception: return pd.Series([np.nan, np.nan, np.nan])


# ==========================================
# CORE PROCESSING LOGIC
# ==========================================

def preprocess_data(df, training_dataset=False):
    """
    Core engine that handles both training and inference preprocessing entirely in memory.
    """

    df = df.copy() 
    new_cols = {}
    
    SCHEMA_PATH = f"{ARTIFACT_DIR}/master_schema.json"
    encoders_path = f"{ARTIFACT_DIR}/categorical_encoders.pkl"
    scaler_path = f"{ARTIFACT_DIR}/minmax_scaler.pkl"
    scaler_cols_path = f"{ARTIFACT_DIR}/scaler_cols.json"
    medians_path = f"{ARTIFACT_DIR}/medians.json" 

    # 1. Basic Label Encoding (Target) - Only relevant if Label exists
    if "Label" in df.columns:
        df["Label"] = df["Label"].map({"Evil_Twin": 1, "Normal": 0})

    # 2. MAC & Behavioral Specific Features
    if "wlan.da" in df.columns:
        new_cols["is_broadcast"] = (df["wlan.da"] == "ff:ff:ff:ff:ff:ff").astype(float)
        new_cols["is_multicast"] = df["wlan.da"].apply(is_multicast)
    if "frame.time_delta" in df.columns:
        new_cols["packet_rate"] = (1 / pd.to_numeric(df["frame.time_delta"], errors='coerce')).replace([np.inf, -np.inf], 0)
    if "wlan.tag" in df.columns:
        new_cols["wlan_tag_count"] = df["wlan.tag"].apply(lambda x: len(str(x).split("-")) if pd.notna(x) else 0)
    if "dns.qry.name" in df.columns:
        new_cols["dns_qry_is_local"] = df["dns.qry.name"].astype(str).str.contains("local", case=False, na=False).astype(float)

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
    if training_dataset:
        encoders = {}
    else:
        encoders = joblib.load(encoders_path)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            cleaned_col = df[col].apply(lambda x: str(x).split('-')[0].lower() if pd.notna(x) else "unknown").to_frame()
            
            if training_dataset:
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                new_cols[col + '_code'] = enc.fit_transform(cleaned_col).flatten()
                encoders[col] = enc
            else:
                if col in encoders:
                    new_cols[col + '_code'] = encoders[col].transform(cleaned_col).flatten()
                else:
                    new_cols[col + '_code'] = -1 
                    
    if training_dataset:
        joblib.dump(encoders, encoders_path)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # 6. Drop Columns
    drop_targets = COLS_TO_DROP + CATEGORICAL_COLS + MULTI_COLS + invariant_cols
    df.drop(columns=[c for c in drop_targets if c in df.columns], inplace=True, errors="ignore")

    # 7. Type Coercion & Imputation
    for col in df.columns:
        if col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' in num_cols:
        num_cols.remove('Label')

    if training_dataset:
        medians = df[num_cols].median().fillna(0).to_dict()
        with open(medians_path, 'w') as f:
            json.dump(medians, f)
    else:
        with open(medians_path, 'r') as f:
            medians = json.load(f)

    for col in num_cols:
        fill_val = medians.get(col, 0)

        if pd.isna(fill_val): 
            fill_val = 0
        df[col] = df[col].fillna(fill_val)

    # 8. Consistent Normalization
    if training_dataset:
        cols_to_normalize = [c for c in df.columns if c != 'Label' and df[c].nunique() > 2]
        scaler = MinMaxScaler()
        if cols_to_normalize:
            df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        
        joblib.dump(scaler, scaler_path)
        with open(scaler_cols_path, 'w') as f:
            json.dump(cols_to_normalize, f)
    else:
        scaler = joblib.load(scaler_path)
        with open(scaler_cols_path, 'r') as f:
            cols_to_normalize = json.load(f)
            
        valid_cols = [c for c in cols_to_normalize if c in df.columns]
        
        # NOTE: reindex uses the schema of the base processed dataset to ensure newly processed datasets have identical columns and order
        if valid_cols:
            temp_df = df.reindex(columns=cols_to_normalize, fill_value=0)
            scaled_data = scaler.transform(temp_df)
            scaled_df = pd.DataFrame(scaled_data, columns=cols_to_normalize, index=df.index)
            for c in valid_cols:
                df[c] = scaled_df[c]

    # 9. Force Master Schema Shape
    if training_dataset:
        master_cols = df.columns.tolist()
        with open(SCHEMA_PATH, 'w') as f:
            json.dump(master_cols, f)
    else:
        with open(SCHEMA_PATH, 'r') as f:
            master_cols = json.load(f)
        
        df = df.reindex(columns=master_cols, fill_value=0)

    return df

# ==========================================
# PUBLIC API FUNCTIONS
# ==========================================

def preprocess_for_training(raw_df):
    """Call this on the training dataset to fit scalers/encoders and define the schema."""
    return preprocess_data(raw_df, training_dataset=True)

def preprocess_for_inference(raw_df):
    """Call this on real-time packet data."""
    return preprocess_data(raw_df, training_dataset=False)


if __name__ == "__main__":
    INPUT_PATH = "data/processed/data.csv"
    OUTPUT_PATH = "data/processed/processed_data.csv"
    
    print(f"Loading training data from {INPUT_PATH}...")
    raw_df = pd.read_csv(INPUT_PATH, low_memory=False)
    
    processed_df = preprocess_for_training(raw_df)
    
    processed_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Processing complete. Data saved to {OUTPUT_PATH}")
    print(f"Final shape: {processed_df.shape}")