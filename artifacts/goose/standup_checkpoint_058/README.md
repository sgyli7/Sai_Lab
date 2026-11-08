# Goose stand-up experimental ONNX — checkpoint 058

`actor_task_jaw.onnx` is the checkpoint 058 task-jaw decoder candidate, based on the checkpoint 252 update-2 actor. SHA-256: `7dbedb732b5191be4921928233d071de1947555e98c99288d37c6a80504f161b`.

This is an experimental stand-up model, not a trained picking or dragging model. Three cold-start fallen-state trials failed (0/3); no robust actor or deployment qualification has been established. The decoder changes the jaw output in the ONNX; it does not represent a newly trained model. The source robot, physics, observation and action contracts are required to evaluate it.

Source: `goose_standup_task_jaw_decoder_256/actor_task_jaw.onnx`; evidence: `goose_standup_checkpoint_058/report.md` and `Bevy_Sim2Sim/docs/goose_training_plan.md` in the experiment workspace. This snapshot preserves an experimental artifact and makes no claim of compatibility with the later Goose 04 hardware model.
