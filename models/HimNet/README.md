# HimNet framework adapter

The recurrent graph architecture follows `HimNet_ori.py`. The adapter accepts
`batch['X']` with channels `[standardized demand, time_of_day, day_of_week / 7]`.
It converts weekdays to integer indices for both the original encoder inputs and
the weekday embedding. TOD embeddings use `86400 / time_intervals` entries.
Linear layers retain the original model's default initialization.

Future decoder covariates contain `[time_of_day, integer day_of_week]`. They are
computed only from the last historical calendar features and a fixed daily
schedule. The training runner infers that schedule from training timestamps and
saves it as `himnet_time_slots` in the experiment config. This supports daily
operating hours that skip nighttime; it assumes the same schedule every day.
Direct construction defaults to a full day; pass `time_slots` for a restricted
schedule. Reconstruct checkpoints with their saved config, including this field.

During training, the engine passes normalized targets and a batch counter that
continues across epochs. Scheduled sampling uses the previous target, never the
current target, as decoder feedback. Validation/test remain autoregressive and
receive only history. The loss is still computed in original demand units by the
shared engine. Standalone `calculate_loss` expects normalized inputs and labels
and returns normalized MAE.

CLI switches: `--use_teacher_forcing 0` disables scheduled sampling;
`--tf_decay_steps 4000` sets its decay scale in batches (the default).

These changes require retraining to compare results. Previous 30-minute models
used a 288-row TOD embedding; the corrected model uses 48 rows, so those state
dicts cannot load directly. No automatic migration of old weights is performed.
Dataset splits and other models' training paths are unchanged.
