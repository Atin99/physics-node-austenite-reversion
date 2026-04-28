# My Project Defense Q-Bank

This is my personal viva/help-pad for the project:

**Physics-Constrained Neural ODE for Austenite Reversion Kinetics in Medium-Mn Steels**

I wrote these notes in first person so I can directly use them in a viva, report discussion, or project explanation. The idea is simple:

- what challenge someone may raise
- how I would answer it
- what limitation I admit
- how I would improve it

## 1. What is my project?

My project is a physics-informed machine learning framework for predicting austenite reversion / transformation behavior in medium-Mn steels under different alloy compositions and annealing conditions.

In simple words, I am trying to do this:

- take inputs like `Mn`, `C`, `Al`, `Si`, temperature, and time
- predict how the transformation curve evolves
- estimate the final metallurgical response

My goal is not to replace experiments completely. My goal is to reduce trial-and-error and build a decision-support model that can help narrow down promising alloy and process conditions.

## 2. My one-line project description

If I have to explain the project in one line, I would say:

> I built a physics-informed Neural ODE framework to predict transformation kinetics in medium-Mn steels using sparse literature data, synthetic augmentation, and real-data-based validation.

## 3. Why did I choose medium-Mn steel?

### Challenge
"Why did you restrict the project to medium-Mn steel when the dataset is already small?"

### My answer
I chose medium-Mn steel because I wanted a metallurgically coherent problem instead of mixing many unrelated steel systems. Medium-Mn steels are strongly connected to annealing-driven retained austenite behavior, so they are a meaningful family for this type of kinetics modeling.

### What I admit
This narrower scope reduces the amount of available data.

### How I would improve it
If I expand the project, I would not expand to all steels blindly. I would only include nearby steel systems whose transformation mechanisms are still comparable, so that I do not dilute the metallurgy.

## 4. Why did I use synthetic data?

### Challenge
"Synthetic data may be wrong. Why did you use it?"

### My answer
I used synthetic data because the real experimental dataset is small, sparse, and heterogeneous. If I trained only on a tiny amount of real data, the model would struggle to learn stable transformation behavior. Synthetic data gave the model broader coverage over composition, temperature, and time space.

### What I admit
Synthetic data is not ground truth. If my synthetic generator is based on weak assumptions, then the model can learn the wrong patterns.

### How I would improve it
- I would always judge the model on held-out real data, not on synthetic data
- I would compare `real-only` and `real + synthetic` runs explicitly
- I would reduce synthetic dominance if real validation becomes worse

## 5. How do I know synthetic data is not hurting the model?

### Challenge
"How do you know synthetic augmentation is not making the project worse?"

### My answer
I do not assume synthetic data is helping automatically. I treat it as a hypothesis. The only valid test is whether the model improves on real validation/test data. If real metrics improve, synthetic data is helping. If real metrics worsen, then synthetic data is hurting.

### What I admit
Because the real dataset is small, this conclusion is still not as strong as I would like.

### How I would improve it
I would run formal ablations:

- `real-only`
- `real + calibrated synthetic`
- `real + calibrated + exploratory synthetic`

Then I would compare `RealRMSE` and `RealEnd` across these settings.

## 6. Why did I use a Neural ODE?

### Challenge
"Why did you use a Neural ODE instead of a simpler regression model?"

### My answer
I used a Neural ODE because I am not only predicting a single endpoint. I am trying to model how the transformation response evolves over time as a curve. A Neural ODE is more suitable for learning time-evolving behavior than a plain static regressor.

### What I admit
Neural ODEs are slower, solver-sensitive, and harder to train than simpler models.

### How I would improve it
I should compare this model against simpler baselines such as:

- an MLP endpoint predictor
- gradient boosting
- a JMAK-based surrogate

If a much simpler model gives similar performance, then I should question whether the ODE complexity is justified.

## 7. Why did I make it physics-informed?

### Challenge
"Why not just use pure machine learning?"

### My answer
I made the model physics-informed because metallurgy data is limited and expensive. Pure black-box learning on sparse data can easily overfit or learn physically unreasonable behavior. By adding physics-based structure, I am trying to guide the model toward more realistic transformation trends.

Examples include:

- diffusion-related features
- thermodynamic terms
- monotonicity or bound constraints
- equilibrium-related reasoning

### What I admit
Physics-informed does not automatically mean correct. If my physical approximations are weak, they can also bias the model.

### How I would improve it
I would validate each physics prior separately and clearly state where it is an approximation rather than exact metallurgy.

## 8. What do my metrics mean?

## `RMSE`

- root mean squared error on normalized scale
- mostly useful for numerical training quality
- lower is better

### Challenge
"Low RMSE does not automatically mean good metallurgy."

### My answer
I agree. A low normalized RMSE only says that the model is fitting the target numerically in training/validation space. It does not by itself prove real metallurgical usefulness.

## `RealRMSE`

- root mean squared error on the real target scale
- this is more meaningful than normalized RMSE
- lower is better

### Challenge
"Why do you focus more on RealRMSE?"

### My answer
Because `RealRMSE` reflects actual prediction error in the real target scale, so it is closer to the practical interpretation of model usefulness.

## `End`

- error in final predicted endpoint on normalized scale

## `RealEnd`

- error in final endpoint on real scale
- important because final state often matters directly in metallurgy

## `NFE`

- number of function evaluations used by the ODE solver
- this reflects computational cost

### Challenge
"Why does NFE matter in your project?"

### My answer
Because even if the model is accurate, it becomes impractical if every prediction is too expensive. If I want to screen many compositions or annealing schedules, solver cost matters.

## 9. Is `RealRMSE` around `0.2` good enough?

### Challenge
"Your model still has around 0.2 error. Is that actually good?"

### My answer
If I claim this is a final industrial-grade predictor, then no, it is not strong enough.

If I present it honestly as a research prototype under severe real-data scarcity, then yes, it is still acceptable for a high-level report because it demonstrates a working framework and meaningful trend learning.

### What I admit
A `RealRMSE` around `0.2` can still mean poor prediction for some individual cases, even if the average score looks reasonable.

### How I would improve it
- collect more real data
- use stricter case-wise evaluation
- report per-study and per-alloy errors, not only averages

## 10. Why do I keep the lowest-score checkpoint?

### Challenge
"A higher-RMSE checkpoint may work better for one special case. Why keep only the lowest score?"

### My answer
I keep the lowest validation-score checkpoint because it is the safest default choice on average. I am not claiming it is best for every single alloy or process condition. I am only saying it has the best average validation behavior among the checkpoints.

### What I admit
Single-checkpoint selection can miss useful diversity among nearby good models.

### How I would improve it
Instead of using only one checkpoint, I would keep a small ensemble of the best checkpoints.

Example:

- top `3-5` checkpoints by `val_real_rmse`
- average their predictions
- use their disagreement as an uncertainty signal

## 11. Why don’t I use all checkpoints?

### Challenge
"If different models may work for different cases, why not use all of them?"

### My answer
Because not all checkpoints are equally good. Some are undertrained, noisy, or clearly worse. If I combine every checkpoint blindly, weak models can pull the prediction away from the truth.

### What I admit
Using only one model may underuse useful information.

### How I would improve it
I would use top-k checkpoint ensembling, not all-history ensembling.

## 12. What are my main dataset numbers?

From the current run:

- real experimental points: `227`
- real curves: `105`
- total curves after augmentation: `1055`
- total points after augmentation: `57227`

### Challenge
"Doesn't this mean your model is mostly learning synthetic data?"

### My answer
Yes, that is a real concern. That is why I use real-data weighting and why I care much more about real-data validation than total training loss.

### What I admit
The training distribution is dominated by synthetic points.

### How I would improve it
- increase the amount of real data
- reduce the synthetic ratio
- do provenance-aware ablations

## 13. What do the important config parameters mean?

These are the parameters I should be ready to explain.

## Model / ODE parameters

### `latent_dim = 32`
This is the size of the latent dynamic representation. If I increase it, the model becomes more expressive, but it also becomes easier to overfit.

### `composition_embed_dim = 32`
This is the size of the learned composition embedding. I use it to encode alloy chemistry into a trainable representation.

### `n_attention_heads = 4`
This is the number of attention heads in the composition encoder. The idea is that different heads can learn different element interactions, such as Mn-C or Al-Mn effects.

### `hidden_dims = [128, 128, 96, 64]`
These are the hidden layer sizes. They control the capacity of the networks inside the model.

### `augmented_dim = 4`
This is the number of extra latent dimensions I add to the ODE state. I use augmentation to help the Neural ODE represent more complex dynamics.

### `solver = "dopri5"`
This is the ODE solver I use for integration. I chose it because it is a standard adaptive-step solver for this kind of task.

### `adjoint`
This controls whether adjoint backpropagation is used. It helps save memory, especially for larger runs, but it can also change numerical behavior.

### `rtol`, `atol`
These are solver tolerances. Smaller values give stricter numerical integration, but they also increase computation.

### `max_num_steps`
This caps the solver steps so that integration does not run away or become too slow.

## Training parameters

### `learning_rate`
This controls the optimizer step size. If it is too high, training becomes unstable. If it is too low, training becomes too slow.

### `batch_size`
This controls how many samples I use in one optimization step. It affects GPU memory, stability, and training noise.

### `max_epochs`
This is the maximum number of epochs I allow. More epochs give the model more time to learn, but also increase the chance of overfitting.

### `early_stopping_patience`
This tells me how long I wait before stopping when validation is no longer improving.

### `gradient_clip_val`
This helps prevent exploding gradients and stabilizes training.

### `use_amp`
This controls mixed precision training. It can speed up GPU runs, but I need to watch numerical stability.

## Data parameters

### `real_data_weight`
This gives extra importance to real experimental data in the loss. I use it because real data is the most trusted source.

### `synthetic_calibration_samples`
This is the number of synthetic curves I generate near real endpoints, so the synthetic data stays partially anchored to reality.

### `synthetic_exploration_samples`
This is the number of broader synthetic curves used to give the model wider coverage.

### `real_only`
This allows me to train only on real data. It is important as an honesty check and ablation setting.

## 14. What if someone says, "Your model can still predict garbage"?

### Challenge
"Even with your best score, your model may still predict badly on many cases."

### My answer
That criticism is fair. Average metrics do not guarantee case-wise reliability. I do not treat this model as a replacement for experiments. I treat it as an exploratory screening model whose outputs still need experimental judgment.

### What I admit
With limited real data, my confidence in out-of-distribution predictions is weak.

### How I would improve it
- report per-alloy and per-study errors
- add uncertainty estimates
- identify failure regions and target those for new experiments

## 15. What do I consider the real contribution of this project?

If someone asks me what I actually contributed, I would say:

- I curated literature-derived medium-Mn steel data into a usable dataset
- I built a provenance-aware pipeline combining real and synthetic data
- I used a physics-informed dynamic model rather than only a static black-box fit
- I evaluated the model on real validation data
- I identified the limitations honestly instead of overclaiming performance

## 16. What would I improve next?

My strongest next steps would be:

1. collect more real experimental data
2. run strict synthetic-data ablation studies
3. compare against simpler baselines
4. use top-k checkpoint ensemble instead of only one checkpoint
5. report per-study / per-alloy performance
6. add uncertainty-based decision support

## 17. Best attitude for viva

I should not overclaim.

My best style in viva should be:

- I defend the core idea
- I admit the limitations clearly
- I show that I already understand the weaknesses
- I explain how I would test, improve, or fix them

That usually sounds much stronger than pretending my model is already perfect.
