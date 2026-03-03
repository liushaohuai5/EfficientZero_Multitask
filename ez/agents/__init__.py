from ez.agents.ez_atari import EZAtariAgent
from ez.agents.ez_dmc_image import EZDMCImageAgent
from ez.agents.ez_dmc_state import EZDMCStateAgent
from ez.agents.ez_humanoid_bench_state import EZHumanoidBenchStateAgent
from ez.agents.ez_maniskill_state import EZManiSkillStateAgent

names = {
    'atari_agent': EZAtariAgent,
    'dmc_image_agent': EZDMCImageAgent,
    'dmc_state_agent': EZDMCStateAgent,
    'humanoid_bench_state_agent': EZHumanoidBenchStateAgent,
    'maniskill_state_agent': EZManiSkillStateAgent,
}