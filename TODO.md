# TODO: Debug and Fix Stock Price Prediction Codebase

## Critical Fixes (Runtime Failures)
- [x] Fix model loading path mismatch in `backend/services/predictor.py` (load `transformer_v1.pt` instead of `transformer_champion.pt`)
- [x] Fix hardcoded dates in screener (backend/frontend sync, add date input in frontend, correct last_price filtering)
- [x] Fix drift detection JSON path in `backend/services/drift_checker.py` (update to correct Evidently structure)

## Major Fixes (Logic/Usability Issues)
- [x] Improve frontend drift page reliability (replace sleep with polling in `frontend/pages/Drift.py`)
- [x] Fix requirements.txt version issues (correct PyTorch versions, remove tqdm upper bound)
- [x] Update default prediction date in frontend to today (avoid future date failures)

## Minor Fixes (Code Quality/Edge Cases)
- [x] Add logging config in `backend/main.py`
- [x] Add data length check in backtest router
- [x] Fix progress bar increment in frontend screener

## Testing and Validation
- [x] Test backend startup (model loads without error)
- [x] Test prediction/screener endpoints with dynamic dates
- [x] Test drift detection and report generation
- [x] Test backtest with short date ranges
- [x] Run full docker-compose to verify integration
