# Generalization-Gap Definition

Primary stored loss gap: `validation_NLL - training_NLL` at each epoch; signed, not absolute. The final reported value uses the final training epoch, not the selected checkpoint epoch. An additional accuracy gap is stored as `training_accuracy - validation_accuracy`. The controller's `V_t` is the positive part of the loss gap: `max(0, validation_NLL - training_NLL)`.
