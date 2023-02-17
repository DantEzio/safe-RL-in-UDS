# safe-RL-in-UDS
The source code, data and figures of an under review research about using safe learning in the real-time control of urban drainage systems for flooding mitigation. 
The article is now under consideration of Journal of Hydrology: Flooding Mitigation through Safe & Trustworthy Reinforcement Learning.

## UDS Environment
Astlingen: A benchmark SWMM model Astlingen of a combined sewer system with 6 storage tanks and 6 orifices. Thanks to Dr. Sun and other contributors for developing this model. More details can be found in following researches:

- Sun C, Lorenz Svensen J, Borup M, Puig V, Cembrano G, Vezzaro L. An MPC-Enabled SWMM Implementation of the Astlingen RTC Benchmarking Network. Water. 2020; 12(4):1034. https://doi.org/10.3390/w12041034
- Schütze, M.; Lange, M.; Pabst, M.; Haas, U. Astlingen—A benchmark for real time control (RTC). Water Sci. Technol. 2017, 2, 552–560. https://doi.org/10.2166/wst.2018.172
- Zhang, Z., Tian, W., & Liao, Z., 2023. Towards coordinated and robust real-time control: A decentralized approach for combined sewer overflow and urban flooding reduction based on multi-agent reinforcement learning. Water Research, 229, 119498. https://doi.org/10.1016/j.watres.2022.119498

## RL Algorithm
- DQN: Deep Q-learning
- PPO: Proximal policy optimization

## Requirements:
- tensorflow 1.14.0
- pyswmm 0.6.0 (have to)
