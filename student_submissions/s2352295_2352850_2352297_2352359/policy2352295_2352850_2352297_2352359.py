from policy import Policy
from student_submissions.s2352295_2352850_2352297_2352359 import algorithm

class Policy2352295_2352850_2352297_2352359(Policy):
    def __init__(self, policy_id=1):
        # Student code here
        
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy=algorithm.BottomLeft()
        elif policy_id == 2:
            self.policy=algorithm.FFDH()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed