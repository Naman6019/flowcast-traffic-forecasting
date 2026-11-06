# Decision tree rules (depth 3)

## Next-window volume
```
|--- vol_profile <= 511.24
|   |--- vol_profile <= 293.59
|   |   |--- vol_t <= 183.50
|   |   |   |--- value: [143.35]
|   |   |--- vol_t >  183.50
|   |   |   |--- value: [227.38]
|   |--- vol_profile >  293.59
|   |   |--- vol_profile <= 397.19
|   |   |   |--- value: [336.02]
|   |   |--- vol_profile >  397.19
|   |   |   |--- value: [438.60]
|--- vol_profile >  511.24
|   |--- vol_profile <= 826.84
|   |   |--- vol_ratio_mean4 <= 0.81
|   |   |   |--- value: [440.55]
|   |   |--- vol_ratio_mean4 >  0.81
|   |   |   |--- value: [663.66]
|   |--- vol_profile >  826.84
|   |   |--- vol_ratio_mean4 <= 0.82
|   |   |   |--- value: [686.59]
|   |   |--- vol_ratio_mean4 >  0.82
|   |   |   |--- value: [1011.32]

```

## Congestion class
```
|--- speed_profile <= 43.69
|   |--- speed_profile <= 30.40
|   |   |--- speed_profile <= 22.67
|   |   |   |--- class: 3
|   |   |--- speed_profile >  22.67
|   |   |   |--- class: 2
|   |--- speed_profile >  30.40
|   |   |--- event_flag <= 0.50
|   |   |   |--- class: 1
|   |   |--- event_flag >  0.50
|   |   |   |--- class: 2
|--- speed_profile >  43.69
|   |--- vc_t <= 0.48
|   |   |--- speed_profile <= 45.15
|   |   |   |--- class: 0
|   |   |--- speed_profile >  45.15
|   |   |   |--- class: 0
|   |--- vc_t >  0.48
|   |   |--- event_flag <= 0.50
|   |   |   |--- class: 0
|   |   |--- event_flag >  0.50
|   |   |   |--- class: 1

```
