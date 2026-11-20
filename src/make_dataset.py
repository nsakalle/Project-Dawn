"""Writes the reference dataset for the pilot: one night of enterprise batch processing.

The data is synthetic but modelled on a typical nightly batch window (20:00 to 08:00) at a
mid-size company: finance close, risk, retail sales, data warehouse, ML, supply chain,
payroll and IT operations, sharing 10 compute nodes and 3 software licences.

Run:  python src/make_dataset.py        -> writes the 7 CSV files in data/
"""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

NODES = [
    # node_id, node_name, pool, datacenter, cpu_cores, memory_gb, gpu_count, speed_factor, cost_per_hour_usd, available_from, available_to
    ("GEN-01", "General compute 1", "general", "DC-East", 32, 128, 0, 1.00, 1.60, "20:00", "08:00"),
    ("GEN-02", "General compute 2", "general", "DC-East", 32, 128, 0, 1.00, 1.60, "20:00", "08:00"),
    ("GEN-03", "General compute 3", "general", "DC-West", 32, 128, 0, 1.00, 1.60, "20:00", "08:00"),
    ("GEN-04", "General compute 4 (older)", "general", "DC-West", 32, 128, 0, 1.25, 1.10, "20:00", "08:00"),
    ("HMEM-01", "High-memory 1", "highmem", "DC-East", 48, 512, 0, 1.00, 4.20, "20:00", "08:00"),
    ("HMEM-02", "High-memory 2", "highmem", "DC-West", 48, 512, 0, 1.00, 4.20, "20:00", "08:00"),
    ("GPU-01", "GPU server 1 (older)", "gpu", "DC-East", 32, 256, 4, 1.20, 7.50, "20:00", "08:00"),
    ("GPU-02", "GPU server 2", "gpu", "DC-East", 32, 256, 4, 0.80, 9.80, "20:00", "08:00"),
    ("DB-01", "Database batch host", "db", "DC-East", 64, 256, 0, 1.00, 3.50, "20:00", "08:00"),
    ("XFER-01", "File transfer gateway", "transfer", "DC-East", 8, 32, 0, 1.00, 0.40, "20:00", "08:00"),
]

MAINTENANCE = [
    # maint_id, node_id, start, end, reason
    ("MW-01", "DB-01", "02:00", "02:45", "Storage firmware patch"),
    ("MW-02", "GEN-02", "23:00", "23:30", "Monthly OS patch reboot"),
    ("MW-03", "GPU-01", "04:30", "05:30", "GPU driver update"),
    ("MW-04", "XFER-01", "06:30", "06:45", "Certificate rotation"),
]

LICENSES = [
    # license_id, product, concurrent_seats, note
    ("LIC-SAS", "SAS analytics server", 2, "Two concurrent sessions on the current contract"),
    ("LIC-INFA", "Informatica ETL", 3, "Three concurrent workflows"),
    ("LIC-TAB", "Tableau extract refresh", 2, "Two concurrent extract refreshes"),
]

EXTERNAL = [
    # ext_id, description, source_system, ext_type, expected_time, latest_time, owner_team, contact
    ("EXT-HR-TS", "Timesheet export from HR system", "HR SaaS", "file_arrival", "20:30", "21:30", "HR Systems", "hr-oncall"),
    ("EXT-DB-SNAP", "Storage snapshot of orders database complete", "Storage array", "upstream_batch", "20:45", "21:15", "IT Operations", "storage-oncall"),
    ("EXT-POS-EU", "EU store point-of-sale files", "Store systems EU", "file_arrival", "21:00", "22:00", "Retail IT", "retail-eu-oncall"),
    ("EXT-ERP-CLOSE", "ERP daily ledger close signal", "ERP", "upstream_batch", "21:30", "22:30", "Finance Systems", "erp-oncall"),
    ("EXT-CRM-API", "CRM bulk export window opens", "CRM SaaS", "api_window", "22:00", "22:00", "Sales Ops", "crm-admin"),
    ("EXT-FX", "End-of-day FX rates file", "Market data vendor", "file_arrival", "22:15", "23:00", "Treasury", "treasury-desk"),
    ("EXT-WEATHER", "Weather forecast feed", "Weather API", "api_window", "22:30", "23:30", "Data Science", "ds-oncall"),
    ("EXT-MKT", "Closing prices and positions feed", "Market data vendor", "file_arrival", "23:30", "00:30", "Risk Analytics", "risk-oncall"),
    ("EXT-SUPP-EDI", "Supplier inventory EDI messages", "EDI gateway", "file_arrival", "00:00", "01:00", "Supply Chain Systems", "edi-oncall"),
    ("EXT-WEBLOGS", "Web server log rotation complete", "Web platform", "upstream_batch", "00:05", "00:20", "Web Platform", "web-oncall"),
    ("EXT-POS-US", "US store point-of-sale files", "Store systems US", "file_arrival", "01:00", "02:00", "Retail IT", "retail-us-oncall"),
    ("EXT-BANK-WIN", "Bank SFTP intake window opens", "Bank", "api_window", "02:00", "02:00", "Treasury", "treasury-desk"),
]

# job_id, job_name, application, owner_team, job_type, priority, cpu_cores, memory_gb, gpu_count,
# est_runtime_min, runtime_p90_min, allowed_pools, license_id, earliest_start, deadline, deadline_type, external_deps
G, H, GH = "general", "highmem", "general;highmem"
JOBS = [
    ("FIN01", "GL extract from ERP", "Finance close", "Finance Data", "etl", "P1", 8, 32, 0, 45, 60, GH, "LIC-INFA", "20:00", "02:00", "soft", "EXT-ERP-CLOSE"),
    ("FIN02", "AP subledger load", "Finance close", "Finance Data", "etl", "P2", 8, 24, 0, 30, 40, GH, "LIC-INFA", "20:00", "03:00", "soft", "EXT-ERP-CLOSE"),
    ("FIN03", "AR subledger load", "Finance close", "Finance Data", "etl", "P2", 8, 24, 0, 35, 45, GH, "LIC-INFA", "20:00", "03:00", "soft", "EXT-ERP-CLOSE"),
    ("FIN04", "FX rates load", "Finance close", "Finance Data", "etl", "P1", 2, 4, 0, 10, 15, G, "", "20:00", "01:00", "soft", "EXT-FX"),
    ("FIN05", "Intercompany reconciliation", "Finance close", "Finance Data", "compute", "P1", 16, 96, 0, 60, 85, H, "LIC-SAS", "20:00", "04:00", "soft", ""),
    ("FIN06", "Trial balance build", "Finance close", "Finance Data", "compute", "P1", 12, 64, 0, 40, 55, GH, "", "20:00", "05:00", "hard", ""),
    ("FIN07", "Group consolidation", "Finance close", "Finance Data", "compute", "P1", 16, 128, 0, 50, 70, H, "LIC-SAS", "20:00", "05:30", "hard", ""),
    ("FIN08", "Management pack PDF render", "Finance close", "Finance Data", "report", "P2", 4, 16, 0, 20, 30, G, "", "20:00", "06:30", "soft", ""),
    ("FIN09", "Finance report distribution", "Finance close", "Finance Data", "file_transfer", "P2", 1, 2, 0, 10, 15, "transfer;general", "", "20:00", "07:00", "hard", ""),
    ("RSK01", "Positions load", "Risk", "Risk Analytics", "etl", "P1", 8, 32, 0, 25, 35, GH, "LIC-INFA", "20:00", "03:00", "soft", "EXT-MKT"),
    ("RSK02", "Market data cleanse", "Risk", "Risk Analytics", "compute", "P2", 4, 16, 0, 20, 30, G, "", "20:00", "03:00", "soft", "EXT-MKT"),
    ("RSK03", "Value-at-Risk calculation", "Risk", "Risk Analytics", "compute", "P1", 32, 256, 0, 90, 130, H, "LIC-SAS", "20:00", "05:00", "hard", ""),
    ("RSK04", "Stress test scenarios", "Risk", "Risk Analytics", "compute", "P2", 24, 192, 0, 75, 100, H, "LIC-SAS", "20:00", "06:00", "soft", ""),
    ("RSK05", "Counterparty exposure", "Risk", "Risk Analytics", "compute", "P2", 16, 64, 0, 45, 60, GH, "", "20:00", "06:00", "soft", ""),
    ("RSK06", "Regulatory risk report", "Risk", "Risk Analytics", "report", "P1", 4, 16, 0, 25, 35, G, "", "20:00", "06:30", "hard", ""),
    ("SLS01", "POS ingest EU", "Retail sales", "Retail Analytics", "etl", "P2", 8, 16, 0, 30, 45, G, "", "20:00", "02:00", "soft", "EXT-POS-EU"),
    ("SLS02", "POS ingest US", "Retail sales", "Retail Analytics", "etl", "P2", 8, 16, 0, 40, 60, G, "", "20:00", "04:00", "soft", "EXT-POS-US"),
    ("SLS03", "Sales dedup and cleanse", "Retail sales", "Retail Analytics", "compute", "P2", 16, 64, 0, 35, 50, GH, "", "20:00", "05:00", "soft", ""),
    ("SLS04", "Product dimension refresh", "Retail sales", "Retail Analytics", "etl", "P3", 4, 16, 0, 15, 20, G, "LIC-INFA", "20:00", "04:00", "soft", ""),
    ("SLS05", "Sales fact load", "Retail sales", "Retail Analytics", "etl", "P2", 16, 64, 0, 45, 60, GH, "LIC-INFA", "20:00", "05:30", "soft", ""),
    ("SLS06", "Daily sales aggregates", "Retail sales", "Retail Analytics", "compute", "P2", 8, 32, 0, 25, 35, G, "", "20:00", "06:00", "soft", ""),
    ("SLS07", "Sales dashboard refresh", "Retail sales", "Retail Analytics", "report", "P2", 4, 8, 0, 20, 30, G, "LIC-TAB", "20:00", "07:00", "hard", ""),
    ("SLS08", "Store performance dashboard", "Retail sales", "Retail Analytics", "report", "P3", 4, 8, 0, 15, 25, G, "LIC-TAB", "20:00", "07:30", "soft", ""),
    ("DW01", "CRM accounts extract", "Data warehouse", "Data Platform", "etl", "P3", 4, 8, 0, 30, 50, G, "", "20:00", "03:00", "soft", "EXT-CRM-API"),
    ("DW02", "CRM opportunities extract", "Data warehouse", "Data Platform", "etl", "P3", 4, 8, 0, 25, 40, G, "", "20:00", "03:00", "soft", "EXT-CRM-API"),
    ("DW03", "HR employee extract", "Data warehouse", "Data Platform", "etl", "P3", 2, 8, 0, 15, 20, G, "", "20:00", "03:00", "soft", "EXT-HR-TS"),
    ("DW04", "Core warehouse load", "Data warehouse", "Data Platform", "etl", "P2", 24, 128, 0, 60, 80, GH, "LIC-INFA", "20:00", "05:00", "soft", ""),
    ("DW05", "Warehouse index rebuild", "Data warehouse", "Data Platform", "maintenance", "P3", 32, 128, 0, 40, 60, "db", "", "20:00", "06:00", "soft", ""),
    ("DW06", "Optimizer statistics gather", "Data warehouse", "Data Platform", "maintenance", "P3", 16, 64, 0, 25, 35, "db", "", "20:00", "06:30", "soft", ""),
    ("DW07", "Customer 360 mart build", "Data warehouse", "Data Platform", "compute", "P2", 16, 64, 0, 35, 50, GH, "", "20:00", "06:30", "soft", ""),
    ("DW08", "Data quality checks", "Data warehouse", "Data Platform", "compute", "P2", 8, 16, 0, 20, 30, G, "", "20:00", "07:00", "soft", ""),
    ("ML01", "Feature store build", "Machine learning", "Data Science", "compute", "P2", 16, 128, 0, 45, 60, GH, "", "20:00", "05:00", "soft", ""),
    ("ML02", "Weather features", "Machine learning", "Data Science", "etl", "P3", 2, 4, 0, 10, 15, G, "", "20:00", "04:00", "soft", "EXT-WEATHER"),
    ("ML03", "Demand forecast training", "Machine learning", "Data Science", "ml_training", "P2", 16, 128, 2, 90, 120, "gpu", "", "20:00", "05:30", "soft", ""),
    ("ML04", "Churn model training", "Machine learning", "Data Science", "ml_training", "P3", 16, 96, 2, 75, 100, "gpu", "", "20:00", "06:30", "soft", ""),
    ("ML05", "Demand forecast scoring", "Machine learning", "Data Science", "ml_scoring", "P2", 8, 64, 1, 30, 40, "gpu", "", "20:00", "06:00", "soft", ""),
    ("ML06", "Churn scoring", "Machine learning", "Data Science", "ml_scoring", "P3", 8, 48, 1, 25, 35, "gpu", "", "20:00", "07:00", "soft", ""),
    ("ML07", "Fraud model retrain", "Machine learning", "Data Science", "ml_training", "P4", 16, 128, 4, 120, 160, "gpu", "", "20:00", "08:00", "soft", ""),
    ("ML08", "Model results publish", "Machine learning", "Data Science", "file_transfer", "P2", 2, 4, 0, 10, 15, "transfer;general", "", "20:00", "07:00", "soft", ""),
    ("SCM01", "Inventory snapshot", "Supply chain", "Supply Chain Systems", "etl", "P2", 8, 32, 0, 20, 30, "db", "", "22:00", "03:00", "soft", ""),
    ("SCM02", "Supplier EDI ingest", "Supply chain", "Supply Chain Systems", "etl", "P2", 2, 8, 0, 15, 25, "transfer;general", "", "20:00", "03:00", "soft", "EXT-SUPP-EDI"),
    ("SCM03", "Replenishment plan", "Supply chain", "Supply Chain Systems", "compute", "P1", 16, 96, 0, 50, 70, H, "", "20:00", "06:30", "hard", ""),
    ("SCM04", "Purchase orders EDI send", "Supply chain", "Supply Chain Systems", "file_transfer", "P1", 1, 2, 0, 10, 15, "transfer", "", "20:00", "07:00", "hard", ""),
    ("PAY01", "Timesheet import", "Payroll", "HR Systems", "etl", "P1", 4, 8, 0, 15, 25, G, "", "20:00", "23:00", "soft", "EXT-HR-TS"),
    ("PAY02", "Gross-to-net payroll calculation", "Payroll", "HR Systems", "compute", "P1", 16, 64, 0, 60, 80, GH, "", "20:00", "03:00", "hard", ""),
    ("PAY03", "Payroll bank file build", "Payroll", "HR Systems", "compute", "P1", 2, 8, 0, 15, 20, G, "", "20:00", "04:00", "hard", ""),
    ("PAY04", "Bank file SFTP send", "Payroll", "HR Systems", "file_transfer", "P1", 1, 2, 0, 5, 10, "transfer", "", "20:00", "05:00", "hard", "EXT-BANK-WIN"),
    ("PAY05", "Payslip PDF generation", "Payroll", "HR Systems", "report", "P3", 8, 16, 0, 40, 55, G, "", "20:00", "07:30", "soft", ""),
    ("OPS01", "Orders DB full backup", "IT operations", "IT Operations", "backup", "P2", 16, 64, 0, 90, 120, "db", "", "20:00", "05:00", "soft", "EXT-DB-SNAP"),
    ("OPS02", "CRM DB incremental backup", "IT operations", "IT Operations", "backup", "P3", 8, 32, 0, 30, 45, "db", "", "20:00", "06:00", "soft", ""),
    ("OPS03", "File share backup", "IT operations", "IT Operations", "backup", "P3", 8, 16, 0, 60, 90, G, "", "21:00", "06:00", "soft", ""),
    ("OPS04", "Backup verification", "IT operations", "IT Operations", "maintenance", "P3", 4, 16, 0, 30, 40, "db", "", "20:00", "07:00", "soft", ""),
    ("OPS05", "Web log archive", "IT operations", "IT Operations", "file_transfer", "P4", 4, 8, 0, 25, 40, "transfer;general", "", "20:00", "08:00", "soft", "EXT-WEBLOGS"),
    ("OPS06", "Security vulnerability scan", "IT operations", "IT Operations", "security", "P3", 8, 16, 0, 45, 60, G, "", "22:00", "06:00", "soft", ""),
    ("OPS07", "Log analytics indexing", "IT operations", "IT Operations", "compute", "P4", 16, 64, 0, 40, 60, GH, "", "20:00", "08:00", "soft", ""),
    ("OPS08", "Temp space cleanup", "IT operations", "IT Operations", "maintenance", "P4", 2, 4, 0, 10, 15, G, "", "05:00", "08:00", "soft", ""),
]

DEPENDENCIES = [
    # predecessor_id, successor_id, lag_min  (all finish-to-start)
    ("FIN01", "FIN05", 0), ("FIN02", "FIN05", 0), ("FIN03", "FIN05", 0), ("FIN04", "FIN05", 0),
    ("FIN05", "FIN06", 0), ("FIN06", "FIN07", 0), ("FIN07", "FIN08", 0), ("FIN08", "FIN09", 5),
    ("RSK01", "RSK03", 0), ("RSK02", "RSK03", 0), ("FIN04", "RSK03", 0),
    ("RSK01", "RSK04", 0), ("RSK02", "RSK04", 0), ("RSK01", "RSK05", 0),
    ("RSK03", "RSK06", 0), ("RSK04", "RSK06", 0), ("RSK05", "RSK06", 0),
    ("SLS01", "SLS03", 0), ("SLS02", "SLS03", 0), ("SLS03", "SLS05", 0), ("SLS04", "SLS05", 0),
    ("SLS05", "SLS06", 0), ("SLS06", "SLS07", 0), ("SLS06", "SLS08", 0),
    ("DW01", "DW04", 0), ("DW02", "DW04", 0), ("DW03", "DW04", 0), ("DW04", "DW05", 0),
    ("DW05", "DW06", 0), ("DW04", "DW07", 0), ("DW07", "DW08", 0), ("SLS05", "DW08", 0),
    ("SLS05", "ML01", 0), ("DW04", "ML01", 0), ("ML01", "ML03", 0), ("ML02", "ML03", 0),
    ("ML01", "ML04", 0), ("ML03", "ML05", 0), ("ML04", "ML06", 0), ("ML05", "ML08", 0), ("ML06", "ML08", 0),
    ("SCM01", "SCM03", 0), ("SCM02", "SCM03", 0), ("ML05", "SCM03", 0), ("SCM03", "SCM04", 5),
    ("PAY01", "PAY02", 0), ("PAY02", "PAY03", 0), ("PAY03", "PAY04", 5), ("PAY02", "PAY05", 0),
    ("OPS01", "OPS04", 0), ("OPS02", "OPS04", 0), ("OPS05", "OPS07", 0),
]


def write(name, header, rows):
    with open(DATA / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name:<28} {len(rows):>3} rows")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    write("jobs.csv", ["job_id", "job_name", "application", "owner_team", "job_type", "priority", "cpu_cores",
                       "memory_gb", "gpu_count", "est_runtime_min", "runtime_p90_min", "allowed_pools",
                       "license_id", "earliest_start", "deadline", "deadline_type", "external_deps"], JOBS)
    write("job_dependencies.csv", ["predecessor_id", "successor_id", "lag_min"], DEPENDENCIES)
    write("resources.csv", ["node_id", "node_name", "pool", "datacenter", "cpu_cores", "memory_gb", "gpu_count",
                            "speed_factor", "cost_per_hour_usd", "available_from", "available_to"], NODES)
    write("maintenance_windows.csv", ["maint_id", "node_id", "start", "end", "reason"], MAINTENANCE)
    write("licenses.csv", ["license_id", "product", "concurrent_seats", "note"], LICENSES)
    write("external_dependencies.csv", ["ext_id", "description", "source_system", "ext_type", "expected_time",
                                        "latest_time", "owner_team", "contact"], EXTERNAL)
