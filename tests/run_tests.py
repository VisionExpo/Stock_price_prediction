"""
Main script to run all tests
"""

import os
import sys
import unittest
import argparse
import time
import pandas as pd
from datetime import datetime

def run_unit_tests():
    """
    Run all unit tests
    """
    print("\n" + "="*80)
    print("Running Unit Tests")
    print("="*80)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join('tests', 'unit')
    suite = loader.discover(start_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """
    Run all integration tests
    """
    print("\n" + "="*80)
    print("Running Integration Tests")
    print("="*80)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join('tests', 'integration')
    suite = loader.discover(start_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_performance_tests(test_type='basic'):
    """
    Run performance tests
    
    Args:
        test_type (str): Type of performance test to run ('basic', 'hyperparameter', 'all')
    """
    print("\n" + "="*80)
    print("Running Performance Tests")
    print("="*80)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join('tests', 'performance', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Import performance test modules
    sys.path.append('.')
    from tests.performance.test_model_performance import test_model_performance, compare_models
    from tests.performance.test_hyperparameter_tuning import tune_hyperparameters
    
    # Run basic performance test
    if test_type in ['basic', 'all']:
        print("\nRunning basic model performance test...")
        test_model_performance(symbol='AAPL', look_back=60, epochs=50, batch_size=32)
    
    # Run model comparison
    if test_type in ['comparison', 'all']:
        print("\nRunning model comparison test...")
        compare_models(symbols=['AAPL', 'MSFT'], look_backs=[30, 60])
    
    # Run hyperparameter tuning
    if test_type in ['hyperparameter', 'all']:
        print("\nRunning hyperparameter tuning test...")
        tune_hyperparameters(
            symbol='AAPL',
            look_backs=[30, 60],
            lstm_units=[50, 100],
            dropout_rates=[0.2],
            batch_sizes=[32],
            epochs=30
        )
    
    return True

def generate_test_report(unit_success, integration_success, performance_success):
    """
    Generate a test report
    
    Args:
        unit_success (bool): Whether unit tests passed
        integration_success (bool): Whether integration tests passed
        performance_success (bool): Whether performance tests passed
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join('tests', 'performance', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create report
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'unit_tests_passed': unit_success,
        'integration_tests_passed': integration_success,
        'performance_tests_passed': performance_success,
        'overall_success': unit_success and integration_success and performance_success
    }
    
    # Convert to dataframe
    report_df = pd.DataFrame([report])
    
    # Save report
    report_file = os.path.join(results_dir, 'test_report.csv')
    report_df.to_csv(report_file, index=False)
    
    print("\n" + "="*80)
    print("Test Report")
    print("="*80)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Unit Tests: {'PASSED' if unit_success else 'FAILED'}")
    print(f"Integration Tests: {'PASSED' if integration_success else 'FAILED'}")
    print(f"Performance Tests: {'PASSED' if performance_success else 'FAILED'}")
    print(f"Overall: {'PASSED' if report['overall_success'] else 'FAILED'}")
    print(f"Report saved to {report_file}")

def main():
    """
    Main function to run tests
    """
    parser = argparse.ArgumentParser(description="Run tests for Stock Price Prediction")
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--performance', choices=['basic', 'comparison', 'hyperparameter', 'all'],
                        help='Run performance tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Track test results
    unit_success = True
    integration_success = True
    performance_success = True
    
    # Run unit tests
    if args.unit or args.all:
        unit_success = run_unit_tests()
    
    # Run integration tests
    if args.integration or args.all:
        integration_success = run_integration_tests()
    
    # Run performance tests
    if args.performance or args.all:
        performance_type = args.performance if args.performance else 'basic'
        performance_success = run_performance_tests(performance_type)
    
    # Generate report
    generate_test_report(unit_success, integration_success, performance_success)
    
    # Return exit code
    return 0 if (unit_success and integration_success and performance_success) else 1

if __name__ == "__main__":
    sys.exit(main())
