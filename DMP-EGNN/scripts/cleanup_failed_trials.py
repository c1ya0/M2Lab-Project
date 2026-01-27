#!/usr/bin/env python3
"""
Clean up failed trials in Optuna database
Note: This will delete failed trials, but will not affect completed trials
"""

import optuna
import argparse
import sys

def cleanup_failed_trials(storage_url, study_name=None, dry_run=True):
    """Clean up failed trials"""
    try:
        if study_name:
            # Clean up specific study
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            
            print(f"📊 Study: {study_name}")
            print(f"   Total trials: {len(study.trials)}")
            print(f"   Failed trials: {len(failed_trials)}")
            
            if failed_trials:
                if dry_run:
                    print(f"   [DRY RUN] Would delete {len(failed_trials)} failed trials")
                    print(f"   Failed trial numbers: {[t.number for t in failed_trials]}")
                else:
                    # Note: Optuna does not directly support deleting trials
                    # But can be achieved by recreating the study
                    print(f"   ⚠️  Note: Optuna doesn't support direct trial deletion")
                    print(f"   Failed trials will be ignored in future optimization")
        else:
            # Clean up all studies
            study_summaries = optuna.get_all_study_summaries(storage=storage_url)
            
            total_failed = 0
            for summary in study_summaries:
                study = optuna.load_study(study_name=summary.study_name, storage=storage_url)
                failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
                total_failed += len(failed_trials)
                
                print(f"📊 Study: {summary.study_name}")
                print(f"   Failed trials: {len(failed_trials)}")
                if failed_trials:
                    print(f"   Failed trial numbers: {[t.number for t in failed_trials]}")
            
            print(f"\nTotal failed trials: {total_failed}")
            print("\n💡 Suggestion: Failed trials will not affect subsequent optimization, Optuna will automatically skip them")
            print("   No need to manually clean up, just continue running optimization")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up failed Optuna trials")
    parser.add_argument("--storage", default="sqlite:///optuna_edmpnn_results_new/optuna_mod_new.db",
                       help="Optuna storage URL")
    parser.add_argument("--study", help="Specific study name (optional)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Only show trials that would be deleted, do not actually delete")
    
    args = parser.parse_args()
    
    cleanup_failed_trials(args.storage, args.study, args.dry_run)

