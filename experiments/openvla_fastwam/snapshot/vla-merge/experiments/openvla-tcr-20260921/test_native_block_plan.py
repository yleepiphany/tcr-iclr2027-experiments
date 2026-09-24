import unittest
from native_block_plan import block_of,make_plan

class Tests(unittest.TestCase):
    def test_grouping(self):
        self.assertEqual(block_of('backbone.language_model.model.layers.7.mlp.up_proj'),'backbone.language_model.model.layers.7')
        self.assertEqual(block_of('action_head.model.mlp_resnet_blocks.0.ffn.1'),'action_head')
        with self.assertRaises(ValueError):block_of('unknown.fc')
    def test_call_quota(self):
        name='backbone.vision_backbone.featurizer.blocks.0.attn.qkv'
        trace=[{'name':name,'input_shape':[1,20,3],'weight_shape':[9,3]}]*2
        trace+=[{'name':'proprio_projector.fc1','input_shape':[1,8],'weight_shape':[10,8]}]
        plan=make_plan([trace]*4)
        self.assertEqual(plan['linear_count'],2)
        self.assertEqual(plan['planned_rows_per_pass_if_all_linears_solved'],1800)
        with self.assertRaises(ValueError):make_plan([trace]*4,cap=1)
        with self.assertRaises(ValueError):make_plan([trace]*3+[list(reversed(trace))])

if __name__=='__main__':unittest.main()
