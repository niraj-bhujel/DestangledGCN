_base_ = ['./base/pedestrians.py']

model = 'GoalDec'

goal_inputs = ['rel', 'pos', 'vnorm']

goal_loss = 'normal'
goal_loss_wt = 1.0

early_stop_metric = 'test_goal_err'

net = dict(type='RNNDec',
			layer='TransformerLayer',
			hidden_dim=64,
			num_layers=2,
			num_heads=2,
			attn_dropout=0.0,
			)