# PPO code for Procgen (Jax + Flax + Optax)

The repo provides a minimalistic and stable implementation of OpenAI's Procgen PPO algorithm (which is in Tensorflow v.1) in Jax [[link to paper]](https://arxiv.org/abs/1912.01588) [[link to official codebase]](https://github.com/openai/train-procgen).

The code's performance was benchmarked on all 16 games using 3 random seeds (see W&B dashboard).

## Prerequisites & installation

The only packages needed for the repo are jax, jaxlib, flax and optax.

Jax for CUDA can be installed as follows:

```
pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

To install all requirements all at once, run

```
pip install -r requirements.txt
```

## Use [Training]

After installing Jax, simply run

```
python train_ppo.py --env_name="bigfish" --seed=31241 --train_steps=200_000_000
```

to train PPO on bigfish on 200M frames. Right now all returns reported on the W&B dashboard are for training on 200 "easy" levels

and validating both on the same 200 levels as well as the entire "easy" distribution (this can be changed in the `train_ppo.py` file).

## Use [Evaluation]

The code automatically saves the most recent training state checkpoint, which can then be loaded to evaluate the PPO policies. Pre-trained policy weights for 25M frames are included in the repo in the `ppo_weights_25M.zip` zip file.

To evaluate the pre-trained policies, run

```
wget https://www.dropbox.com/s/2jrcf3ebexeppbt/ppo_weights_25M.zip?dl=1
unzip ppo_weights_25M.zip?dl=1
python evaluate_ppo.py --env_name="starpilot" --model_dir="model_weights"
```

## Results

See this W&B report for aggregate performance metrics on all 16 games (easy mode, 200 training levels and all test levels):

[Link to results](https://wandb.ai/bmazoure/ppo_procgen_jax/reports/PPO-Procgen-JAX-version---VmlldzoxMDM4MjAx)

## Cite

To cite this repo in your publications, use

```
@misc{ppojax,
  author = {Mazoure, Bogdan},
  title = {Jax implementation of PPO},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bmazoure/ppo_jax}},
}
```