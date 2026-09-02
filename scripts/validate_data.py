import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.nasa_loader import NASADataLoader
from src.data.calce_loader import CALCEDataLoader
from src.data.validator import DataQualityValidator
from src.utils.logger import setup_logger

logger = setup_logger("validate_data")

def main():
    logger.info("Running Data Quality Validation...")
    nasa_loader = NASADataLoader()
    calce_loader = CALCEDataLoader()
    
    nasa_df = nasa_loader.load_all()
    calce_df = calce_loader.load_all()
    
    validator = DataQualityValidator()
    report = validator.run_full_validation(nasa_df, calce_df)
    
    print("==========================================")
    print("DATA QUALITY VALIDATION SUMMARY")
    print("==========================================")
    print("Overall Status: " + str(report["overall_status"]))
    print("NASA Records: " + str(report["datasets"]["nasa_pcoe"]["total_records"]) + " across cells " + str(report["datasets"]["nasa_pcoe"]["cells"]))
    print("CALCE Records: " + str(report["datasets"]["calce_cs2"]["total_records"]) + " across cells " + str(report["datasets"]["calce_cs2"]["cells"]))
    print("==========================================")

if __name__ == "__main__":
    main()
