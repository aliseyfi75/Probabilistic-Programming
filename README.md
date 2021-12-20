# Probabilistic-Programming

## Assignment 1
- Link to WandB project : https://wandb.ai/aliseyfi/LDA/runs/1ykjf0cn/overview?workspace=user-aliseyfi

## Project

- `bbvi/` : here is bbvi code for `lr=0.05` and `L=5`
    - You have access to wandb where you can have that beautiful plot of ELBO
    - `bbvi.out` you have q after each 25 steps here
    - `save/` you have `elbo_plot`, `posterior_plot`, and `q_plot` plots
- `IS/`: 
    - you have `thetas.pt`  and `weights.pt` which is for `20000` samples
    - `IS.out` results of `20000` samples
    - `Experiment_without_BBVI`: This is the results of starting with the original prior we had
        - `IS_#samples.out`  results of `#samples`
        - `results/` :
            - `MSE`, `thetas`, `weights`, `within_3` results are here, they are all `pytorch` objects that you can load them into code using `pytorch.load()`.
    - `Experiment_with_BBVI`: This is the results of starting with final proposal of BBVI:
        - `IS_#samples.out`  results of #samples
        - `results/` :
            - `MSE`, `thetas`, `weights`, `within_3` results are here, they are all `pytorch` objects that you can load them into code using `pytorch.load()`.
- `MCMC/`: results of `1000` samples
    - `MCMC.out` : final results
    - `figures/` containing `histograms` and `traces` for each theta
- `smc_IS` : This is our implementation of SMC using IS
    - `Experiment_without_BBVI`: This is the results of starting with the original prior we had
        - `smc_#particles.out`  results of `#particles`
        - `results`:
            - `MSE`, `thetas`, `weights`, `within_3` , and `logZ` results are here, they are all `pytorch` objects that you can load them into code using `pytorch.load()`.
    - `Experiment_with_BBVI`: This is the results of starting with the original prior we had
        - `smc_#particles.out`  results of `#particles`
        - `results`:
            - `MSE`, `thetas`, `weights`, `within_3` , and `logZ` results are here, they are all `pytorch` objects that you can load them into code using `pytorch.load()`.