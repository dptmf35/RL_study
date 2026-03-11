# Repository Guidelines

## Project Structure & Module Organization
`applications/` contains Isaac Sim entrypoints (`spot_warehouse.py`, `spot_warehouse_dashboard.py`, `spot_nav_demo.py`) and policy wrappers.  
`dashboard/backend/` hosts the FastAPI + WebSocket bridge and HDF5 recorder.  
`dashboard/frontend/` is a React + TypeScript (Vite) UI.  
`assets/` stores USD/USDA robot models and textures.  
`policies/` stores pretrained `.pt` policies and env params.  
`data/recordings/` stores generated `session_YYYYMMDD_HHMMSS.h5` files and should be treated as runtime output.

## Build, Test, and Development Commands
- `~/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh -m pip install fastapi "uvicorn[standard]" h5py websockets`: install backend/runtime deps in Isaac Sim Python.
- `cd dashboard/frontend && npm install`: install dashboard UI deps.
- `./python.sh /home/yeseul/IsaacRobotics/applications/spot_warehouse.py`: run base teleop app.
- `./python.sh /home/yeseul/IsaacRobotics/applications/spot_warehouse_dashboard.py`: run teleop + data collection backend.
- `cd dashboard/frontend && npm run dev`: start dashboard frontend (`http://localhost:5173`).
- `cd dashboard/frontend && npm run build`: type-check and build production frontend bundle.

## Coding Style & Naming Conventions
Python: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, type hints for new public functions.  
TypeScript/React: functional components, `PascalCase` component files, hooks under `src/hooks` with `useX` naming, strict TS settings must pass.  
Keep modules focused; separate simulation control, API transport, and recording logic.

## Testing Guidelines
There is no formal automated test suite yet. Treat validation as required smoke testing:
- Run target Isaac app and verify no startup errors.
- For dashboard changes, verify `/api/status`, WebSocket live updates, and start/stop recording path.
- For frontend changes, run `npm run build` to catch TS regressions.
Name future tests by behavior (`test_record_stop_without_start`, `useStatus.polls_api`).

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects (for example: `add ros2 publisher`, `update readme`). Follow the same style and keep subject lines concise.  
PRs should include:
- clear scope and affected paths,
- runtime validation steps performed,
- linked issue (if any),
- screenshots/GIFs for dashboard UI changes.
