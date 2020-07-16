ETA=1
RHO=0.001
EPSILON=0.05
NRUNS=30
NEPISODES=500
python main.py --baseline --discount=0.99 --epsilon=$EPSILON \
               --lr_critic=0.25 --lr_intra=0.25 --lr_term=0.25 \
               --nruns=$NRUNS --nsteps=10000 --nepisodes=$NEPISODES \
               --env_id="ConstFourrooms-v0" --eta=$ETA --rho=$RHO \
               --id="subgoal-based-potential" --subgoal-path="in/subgoals/fourrooms_subgoals_1.csv"