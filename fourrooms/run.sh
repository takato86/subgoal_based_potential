ETA=10
RHO=0
EPSILON=0.05
NRUNS=100
NEPISODES=1000
python main.py --baseline --discount=0.99 --epsilon=$EPSILON \
               --lr_critic=0.25 --lr_intra=0.25 --lr_term=0.25 \
               --nruns=$NRUNS --nsteps=10000 --nepisodes=$NEPISODES \
               --env_id="ConstFourrooms-v0" --eta=$ETA --rho=$RHO \
               --id="actor-critic" --subgoal-path="in/subgoals/fourrooms_best_subgoals.csv"