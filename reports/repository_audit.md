# Repository audit

| Item | Finding |
|---|---|
| DT model | `TrainableDT` (CNN occupancy encoder + HF `DecisionTransformerModel`) in `uncertaintyDT/CNNDT_baseline/HF-DT-testing_average_return.py`; imported, not copied |
| Checkpoint | `uncertaintyDT/CNNDT_baseline/output2/model.safetensors` (sha256 prefix `2eb193f8b98521da`), strict=False load: 0 missing / 0 unexpected keys. This is the plain (non-uncertainty-weighted) DT; UW-DT was not used |
| Env | `roundabout-v0` from the local `CNNDT_baseline/highway_env` (symlinked) |
| Observation | OccupancyGrid, features presence/vx/vy/on_road, shape (4, 50, 41), grid +-50 m x +-41 m, 2 m cells, ego-relative |
| Actions | DiscreteMetaAction: 0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT, 3 FASTER, 4 SLOWER; target speeds [0, 8, 16] m/s |
| Frequency | policy_frequency 2 Hz (one DT step = 0.5 s), simulation 15 Hz, duration 11 s, MAX_EP_LEN 22 |
| DT inference | `get_action` (argmax, history cropped to max_length), RTG start MAX_EP_LEN/SCALE = 2.2, SCALE 10 |
| Exit success | route empty and ego not crashed (same test as `UWDT/UADT/entropyandexit.py`); I additionally require no collision in the episode |
| Traffic density | 0-4 incoming vehicles drawn with the *global* numpy RNG, so the scripts call `np.random.seed(seed)` before `reset(seed=seed)`; B0 repeated runs are bit-identical (checked) |
| Ego start | (x=2, y=45): already ~45 m from the roundabout centre at t = 0 |
