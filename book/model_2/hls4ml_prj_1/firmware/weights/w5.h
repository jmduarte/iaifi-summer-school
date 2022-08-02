//Numpy array shape [64, 32]
//Min -0.581462025642
//Max 0.501357138157
//Number of zeros 0

#ifndef W5_H_
#define W5_H_

#ifndef __SYNTHESIS__
fc2_weight_t w5[2048];
#else
fc2_weight_t w5[2048] = {0.146504, 0.057319, -0.069084, -0.000356, -0.236796, -0.000175, -0.000403, -0.077498, -0.410892, -0.062992, -0.002243, -0.140001, -0.017288, 0.338979, -0.045308, 0.178863, -0.263274, -0.148241, -0.010503, 0.000913, 0.120120, -0.439561, -0.059402, 0.029469, -0.316433, 0.205083, -0.208994, -0.223699, -0.410722, -0.407357, -0.148545, 0.118535, 0.091145, 0.173096, 0.130129, -0.000334, -0.000671, 0.029177, -0.060331, 0.001649, -0.124418, -0.207650, -0.000437, 0.031517, -0.074978, 0.061550, 0.002414, 0.196298, -0.048815, 0.101601, 0.063554, -0.000099, 0.058399, 0.202843, 0.117453, -0.118832, 0.080157, 0.175115, 0.001178, -0.000065, 0.091603, -0.048878, -0.000113, 0.000129, 0.132493, -0.126375, -0.086439, -0.000476, -0.126799, -0.074924, -0.000696, 0.091305, -0.297929, 0.000506, 0.070866, 0.084897, 0.122677, 0.117429, 0.009697, -0.031855, 0.127139, 0.010447, 0.116576, 0.000407, -0.131158, 0.202704, 0.025457, -0.055903, 0.006024, 0.203068, -0.075811, -0.173742, -0.048582, -0.090171, -0.102330, -0.000591, 0.124117, 0.125026, 0.003703, -0.108835, 0.097276, -0.329088, -0.142847, 0.093390, -0.050217, 0.016793, -0.223254, -0.126275, -0.086634, -0.076570, 0.191494, 0.000947, -0.097318, -0.089674, 0.000037, -0.062949, 0.071679, -0.091925, -0.032583, -0.060386, -0.054584, -0.211935, 0.027768, -0.217346, -0.066623, -0.014493, -0.153952, -0.051477, -0.050364, -0.060454, 0.101811, -0.000087, 0.008186, 0.028431, -0.001391, 0.248497, 0.146167, 0.289390, -0.076592, 0.154790, 0.251211, -0.210478, 0.187986, 0.000739, -0.049946, -0.015015, 0.120285, -0.014648, 0.244903, 0.344806, 0.120986, -0.047721, 0.229802, -0.080846, 0.210418, 0.146016, 0.084489, 0.088550, 0.000418, 0.002883, 0.217376, 0.200929, 0.095762, 0.000184, -0.000083, 0.019632, 0.001123, 0.004307, 0.007205, -0.132096, 0.141117, 0.152021, 0.103856, 0.013980, -0.058350, -0.107400, 0.341503, 0.183309, -0.113603, -0.018206, 0.118842, -0.004284, -0.229989, 0.056821, -0.210530, 0.091799, -0.000137, 0.058060, 0.174718, 0.079653, -0.003977, -0.125306, 0.257814, -0.220204, 0.089023, -0.000298, 0.001974, -0.000549, -0.122441, 0.012194, -0.033797, -0.041397, -0.000468, 0.073647, 0.027561, -0.069225, 0.218746, 0.235904, -0.274816, -0.000216, -0.167169, -0.100802, 0.117874, -0.011860, 0.191116, -0.002827, -0.062384, 0.127676, -0.049806, -0.000019, -0.111053, -0.023175, -0.022394, -0.052802, -0.204880, -0.282259, 0.057224, 0.039981, -0.000216, 0.047968, 0.038284, 0.034474, 0.061242, 0.309758, 0.001301, 0.011381, 0.135116, -0.100381, 0.090026, -0.044457, 0.067053, -0.000320, -0.124128, 0.001743, 0.068738, 0.018716, 0.019432, -0.138367, -0.015530, 0.251737, 0.000932, 0.112363, 0.041296, 0.002290, -0.002396, 0.097297, 0.273728, 0.159787, -0.095899, -0.082688, 0.044211, -0.239527, -0.052037, -0.044891, -0.034758, 0.240481, -0.200383, 0.224393, 0.088123, 0.110409, 0.027674, -0.156088, 0.189782, -0.060911, 0.000173, 0.000194, -0.014371, 0.302388, 0.074800, -0.126229, 0.102276, -0.101984, 0.155427, -0.059392, 0.053821, -0.151030, -0.145097, -0.133850, -0.272741, -0.045001, -0.001526, -0.000061, 0.043956, 0.099487, 0.170016, 0.167059, 0.159560, -0.119919, 0.110795, 0.011264, -0.065390, 0.007177, 0.161889, -0.013596, 0.216173, 0.133425, -0.004874, 0.000258, 0.001104, -0.226837, 0.044769, -0.040508, -0.189479, 0.102172, 0.061611, 0.131301, 0.155848, 0.276317, 0.324708, 0.000105, -0.100616, -0.003594, 0.213122, 0.000011, -0.001113, -0.062383, -0.001697, -0.131414, -0.039239, -0.073815, -0.139913, -0.119217, -0.042170, 0.171074, 0.150187, 0.295518, -0.186452, -0.106692, 0.120781, 0.034112, -0.121153, 0.093658, 0.151870, 0.108689, 0.100207, 0.098447, -0.093671, -0.174231, -0.192118, -0.078370, -0.000301, 0.181284, 0.177125, 0.112478, -0.111154, -0.045753, 0.000214, 0.139477, -0.003879, 0.117875, -0.005959, 0.090547, -0.143393, 0.163009, -0.128109, -0.070634, 0.159757, 0.243591, -0.047166, -0.022816, 0.000023, 0.000263, 0.106589, -0.123687, -0.058165, -0.027705, 0.147672, -0.193512, 0.034990, -0.000005, -0.115240, -0.000111, 0.000249, 0.050881, -0.033340, 0.202642, 0.288803, -0.024941, -0.020385, -0.342390, 0.016354, 0.208668, -0.125616, -0.397006, -0.195678, 0.064124, 0.002292, -0.111318, 0.013916, 0.000158, 0.210732, -0.042348, 0.017862, -0.085156, -0.027800, 0.410458, -0.008285, 0.309246, 0.246473, -0.030130, 0.000004, -0.244879, -0.000339, -0.375715, -0.147248, 0.049733, 0.106508, -0.073532, -0.036950, 0.161910, -0.001656, -0.073629, 0.081821, 0.160934, 0.209256, 0.003530, -0.044695, 0.005475, -0.006034, -0.133463, 0.213226, -0.134499, 0.028789, 0.071607, 0.125135, 0.000267, 0.124043, -0.061249, 0.060010, -0.105875, -0.026163, -0.000723, 0.000485, 0.136971, 0.063364, -0.030049, 0.036581, 0.014603, 0.050778, -0.056009, 0.195485, 0.000216, -0.000404, 0.000911, -0.087302, 0.231628, 0.040659, 0.004807, -0.251336, -0.048741, -0.039667, 0.260938, 0.000417, 0.074087, 0.046568, 0.070140, 0.068512, 0.078983, -0.051526, 0.161028, 0.047994, 0.150418, 0.000131, 0.106401, -0.080638, -0.169056, 0.102329, -0.259360, -0.014761, 0.104128, -0.076760, 0.133183, -0.038672, 0.000056, -0.057848, 0.000574, 0.042310, 0.177767, -0.098424, -0.048763, -0.212192, -0.053137, -0.003375, -0.023420, 0.035494, 0.202900, -0.329640, -0.066129, 0.131235, 0.000296, 0.027573, -0.097075, 0.065227, 0.035744, -0.013720, 0.186876, 0.092205, -0.164174, -0.255508, -0.090152, -0.000130, 0.000354, -0.189722, 0.085655, 0.192953, 0.000076, -0.000660, -0.280425, 0.033114, 0.177524, -0.202417, -0.036685, -0.000520, 0.016826, -0.074066, -0.009949, 0.208809, -0.063640, -0.145985, 0.000080, 0.182627, 0.000663, -0.151075, -0.055141, 0.003431, 0.014003, -0.000095, 0.039505, -0.093051, -0.002128, 0.000026, -0.192911, -0.000059, -0.120992, 0.171984, 0.256452, 0.083482, 0.102044, -0.289774, -0.226124, -0.000107, 0.325281, -0.162408, 0.009347, -0.035037, -0.000068, -0.191132, 0.033005, -0.000034, 0.058567, 0.207559, -0.000135, -0.065013, 0.012430, -0.009161, 0.170018, 0.280006, 0.092559, 0.009128, -0.069412, -0.139761, -0.188842, 0.000334, -0.009831, 0.000111, 0.013537, -0.120280, 0.154150, 0.037775, 0.000068, 0.202576, 0.073080, -0.000854, 0.080745, 0.247933, -0.012542, 0.009542, 0.082202, 0.233140, -0.184841, -0.033257, -0.190603, -0.091556, -0.086293, -0.171782, -0.082994, 0.061106, 0.200189, 0.029414, 0.106993, -0.030507, -0.000499, -0.102273, 0.219136, 0.099403, -0.060380, 0.211125, -0.000637, 0.094202, 0.036191, 0.065819, -0.000246, 0.170469, -0.080718, 0.045430, 0.212379, -0.029408, -0.096803, -0.109849, -0.091216, -0.178890, -0.008120, -0.057167, 0.197637, -0.001429, 0.029709, 0.139958, 0.005116, 0.002907, -0.040336, 0.165388, 0.230246, 0.192838, 0.177344, 0.055911, -0.030336, -0.000617, -0.075717, 0.000160, 0.169740, 0.022174, -0.082432, 0.107871, -0.000198, -0.092843, 0.010035, 0.146682, 0.073978, 0.256187, -0.174009, 0.013442, -0.018140, -0.002555, 0.206317, -0.030924, -0.149581, -0.000957, 0.005835, -0.005833, 0.000650, -0.021640, -0.158484, -0.095277, 0.162939, 0.001500, 0.160099, 0.051592, -0.001459, 0.114193, 0.012390, 0.020110, 0.064555, -0.157892, 0.103798, 0.107993, -0.000436, 0.161455, -0.108237, -0.010326, 0.080045, 0.024925, -0.004491, 0.104701, 0.111723, 0.054103, 0.146546, 0.077610, -0.225975, 0.007963, 0.173624, 0.122275, 0.227328, -0.034573, 0.112109, -0.249830, -0.013947, -0.011919, -0.091184, -0.120118, -0.060297, 0.036058, 0.073753, -0.001301, 0.125343, -0.330995, -0.247461, -0.017245, -0.000465, -0.216482, 0.310262, 0.190571, -0.015287, -0.062179, -0.346105, 0.002090, -0.140356, -0.210384, 0.158989, 0.000452, 0.054566, -0.011569, -0.021787, -0.126768, 0.113861, -0.000574, -0.440236, -0.013865, 0.198628, -0.227491, 0.068215, 0.000412, -0.139065, 0.042327, 0.210344, 0.016703, 0.166545, -0.142241, 0.115574, 0.148743, -0.000016, -0.157516, 0.127287, 0.014400, 0.074252, -0.001657, -0.070073, -0.139054, -0.114276, 0.140104, -0.180949, -0.009183, -0.019608, -0.060952, -0.225825, 0.027381, -0.001093, 0.042786, 0.181123, -0.139364, 0.086927, 0.037159, -0.174607, 0.140225, -0.050511, -0.113466, -0.183435, -0.002404, -0.160754, 0.352168, 0.113888, -0.054366, -0.000076, 0.248141, 0.028269, 0.101094, 0.070381, 0.039158, 0.220019, 0.134181, 0.023635, 0.216327, -0.016582, 0.108385, -0.001445, 0.054810, -0.000188, -0.008887, 0.000608, 0.161086, 0.283174, -0.204747, -0.081206, 0.105900, -0.147899, -0.094090, 0.163642, 0.052391, 0.069424, -0.164800, -0.012187, -0.232692, 0.000221, 0.118980, 0.000006, 0.225907, -0.044251, -0.295114, 0.165938, 0.270061, -0.225455, 0.294896, 0.000350, 0.397767, -0.258932, -0.151992, -0.144217, 0.060714, 0.310261, -0.199892, -0.006300, 0.071761, 0.310000, -0.092795, -0.000050, 0.065336, 0.102253, 0.080702, 0.254158, -0.003644, 0.133735, 0.134044, -0.003692, -0.054967, -0.292524, 0.151199, 0.000135, -0.058141, -0.006933, -0.023257, 0.113398, -0.038167, -0.361401, 0.027693, 0.053599, -0.293455, 0.235044, 0.003417, -0.013440, 0.301689, 0.137448, 0.149909, 0.191567, 0.088218, -0.097692, 0.080578, 0.246976, 0.240526, 0.170555, -0.007620, -0.023923, 0.244625, 0.143742, 0.001703, 0.075500, 0.144524, 0.084266, -0.004468, 0.000143, 0.271939, -0.014216, -0.123218, -0.014485, 0.205596, 0.100776, 0.152282, 0.000267, 0.113502, -0.220104, -0.029125, -0.098137, 0.430216, 0.414658, 0.100011, -0.101083, 0.331191, 0.026017, -0.164368, 0.198520, -0.008120, -0.084532, 0.194960, 0.214446, 0.350493, 0.196103, 0.001598, -0.089173, -0.369611, -0.239682, 0.113128, 0.005481, -0.027801, 0.322935, 0.052362, -0.017276, -0.124293, -0.066660, -0.003127, 0.013562, -0.121729, 0.395597, -0.046723, -0.004507, 0.215262, -0.006813, -0.007472, 0.083979, 0.006920, -0.242615, 0.085077, 0.149809, -0.106300, -0.099857, 0.000358, 0.032122, 0.123573, 0.198579, 0.230070, 0.072624, -0.321487, -0.118372, 0.234259, -0.000173, 0.066965, 0.000603, 0.185665, -0.040270, -0.010164, -0.077133, 0.143588, 0.002098, -0.048457, 0.001846, 0.101977, -0.011816, 0.020551, 0.252201, -0.031334, 0.000393, -0.008508, 0.062540, -0.000180, -0.031191, 0.007695, -0.029323, 0.066025, -0.002223, 0.162093, 0.116519, 0.205198, 0.000598, -0.011427, -0.066538, -0.006260, -0.000134, 0.000139, 0.059811, 0.015841, 0.000157, -0.081325, 0.075708, -0.060005, 0.236120, -0.215591, 0.039823, -0.110922, -0.078533, -0.044642, 0.006485, -0.008299, 0.156156, 0.156378, -0.080321, -0.196072, 0.086626, -0.007682, -0.091729, -0.177757, 0.003194, 0.098059, 0.232735, 0.128127, 0.228049, -0.007985, 0.184215, -0.012499, -0.000043, 0.107940, 0.075728, -0.250822, -0.000196, 0.217571, 0.204902, -0.100239, 0.050854, 0.132031, -0.213039, 0.392819, 0.161620, -0.323765, -0.304323, -0.041012, 0.000086, 0.000259, 0.005542, -0.242126, -0.322786, -0.124405, -0.074029, 0.062242, 0.501357, -0.000170, 0.030716, 0.104030, 0.000572, 0.407084, 0.244582, -0.122168, -0.058870, -0.036613, -0.006022, -0.021308, -0.018325, -0.357531, 0.212615, -0.224602, 0.014701, -0.014919, 0.119576, 0.095479, 0.137176, -0.025009, -0.089650, 0.201930, -0.034238, 0.164367, 0.094525, -0.056964, -0.179853, -0.027924, -0.026359, -0.032422, -0.164268, -0.286436, -0.232953, -0.040355, -0.234468, -0.099565, 0.090673, 0.057148, -0.000497, 0.029912, -0.042664, -0.102400, 0.000244, -0.100935, -0.035943, -0.066467, -0.002858, -0.020595, -0.076608, 0.181870, 0.070321, 0.108887, -0.056474, 0.146894, 0.000188, -0.111554, -0.128793, 0.094340, 0.023950, 0.066478, -0.143702, 0.227725, -0.000172, 0.111668, -0.008133, -0.000732, -0.019042, 0.146471, 0.049903, -0.184467, 0.000120, 0.200308, 0.012293, 0.024971, -0.264586, -0.048128, 0.166276, 0.042551, 0.065384, -0.111825, -0.171760, 0.008905, -0.023986, 0.027920, 0.001100, 0.007696, 0.030904, 0.049979, -0.155620, 0.068893, -0.013415, -0.117696, -0.186784, -0.026927, 0.001237, 0.000112, -0.179717, -0.030311, 0.040036, 0.277380, 0.172544, -0.209988, 0.000045, 0.051409, 0.165365, -0.188682, -0.057602, 0.098624, 0.222518, 0.085223, 0.060673, 0.163743, -0.506837, 0.013308, 0.158359, -0.133615, -0.250236, -0.258870, -0.121940, 0.038749, -0.095838, -0.548343, -0.100855, -0.269308, -0.054356, 0.317841, 0.442231, -0.003742, 0.094346, 0.175189, -0.118058, 0.075382, 0.006696, -0.000013, -0.004436, 0.024761, -0.004152, -0.009694, -0.069969, -0.171793, 0.107043, -0.065046, -0.002497, 0.044264, 0.250417, 0.056153, -0.033571, -0.140896, -0.228600, -0.160855, -0.045749, -0.148754, -0.234572, 0.104419, 0.082436, 0.059790, 0.006504, -0.030174, -0.262515, -0.217213, -0.114162, -0.425394, -0.018201, -0.014749, 0.045616, -0.102687, 0.000020, -0.041794, 0.179625, -0.039507, 0.023978, -0.000359, -0.057648, -0.077646, 0.138850, 0.018321, 0.006450, 0.166935, 0.090805, -0.164394, -0.094509, -0.090814, -0.000009, -0.138008, 0.000169, -0.148153, 0.028303, 0.102976, 0.000172, -0.052169, 0.040169, -0.039233, -0.031319, 0.000007, 0.000480, -0.109484, 0.176795, -0.140675, -0.101686, 0.001132, 0.164420, 0.176334, -0.000083, -0.019022, 0.154606, -0.001492, 0.022675, 0.096667, 0.046505, 0.244949, 0.057302, -0.057070, -0.060950, -0.017694, 0.002558, -0.018717, -0.038120, 0.168288, 0.006738, 0.003446, -0.005317, 0.000460, -0.032836, -0.015411, 0.016517, 0.114153, 0.181759, 0.064666, -0.109515, 0.101938, 0.000280, -0.074153, 0.020266, -0.001057, 0.000462, 0.113753, 0.186296, 0.056616, 0.033141, 0.108504, -0.109169, 0.250086, 0.000493, -0.001086, 0.134353, -0.063157, 0.007053, -0.037461, 0.023731, 0.139893, 0.071851, 0.197620, 0.029647, -0.002424, 0.133171, 0.001035, -0.041002, 0.000017, 0.048589, 0.206500, 0.258423, -0.121399, -0.094138, -0.000429, -0.184985, 0.050258, 0.156146, -0.127057, 0.150177, -0.149969, 0.008953, 0.126807, -0.038419, 0.225707, -0.015323, 0.027706, -0.031578, 0.056865, 0.000721, 0.075471, 0.273746, 0.020579, 0.108542, 0.282184, 0.150908, 0.045827, -0.016064, -0.109584, -0.015047, 0.114510, 0.002007, -0.056936, 0.097984, 0.000647, 0.000002, -0.000150, 0.269028, 0.001695, -0.144306, -0.000133, -0.069513, -0.029032, -0.021510, 0.018286, -0.127443, 0.095883, -0.001655, -0.079746, -0.165038, 0.035700, -0.009823, 0.076135, 0.100552, -0.015029, 0.178339, 0.266478, -0.123521, -0.079766, -0.000015, -0.004441, -0.011182, -0.000413, 0.000049, 0.146104, -0.031650, -0.184865, 0.000229, 0.134446, 0.112615, -0.251255, 0.002826, 0.125489, -0.070286, 0.006062, 0.165686, -0.000684, 0.228529, 0.053190, 0.134028, -0.091264, 0.021575, 0.087555, 0.065453, 0.067058, -0.010529, 0.002621, -0.036920, 0.091285, 0.165245, 0.068096, 0.056943, -0.000231, 0.130475, 0.000798, 0.207074, 0.210521, 0.271509, -0.006225, -0.000829, 0.136941, 0.092670, -0.404603, 0.159068, 0.191040, 0.183669, 0.136267, 0.282218, 0.339882, -0.581462, -0.089500, 0.161289, 0.000961, -0.000207, -0.215065, -0.190708, -0.122180, -0.114946, -0.450643, -0.136182, -0.005424, 0.232138, 0.221257, 0.353071, 0.064147, 0.036523, 0.349361, -0.060547, 0.045914, 0.115153, -0.000859, 0.000055, 0.293576, 0.030672, -0.050432, 0.041360, 0.231732, 0.214630, 0.226726, 0.304536, 0.433672, -0.462678, -0.067818, -0.005216, 0.009124, -0.110561, -0.198671, -0.079493, 0.189989, 0.213136, -0.240928, 0.218425, -0.044021, -0.151267, 0.190656, 0.386286, -0.074048, -0.007497, 0.166689, -0.151703, 0.117503, 0.055432, 0.012213, 0.000132, 0.000082, 0.071053, 0.048611, 0.042116, -0.002031, -0.033509, 0.070562, 0.002764, -0.205819, 0.404028, 0.004837, 0.219317, 0.111285, -0.121641, 0.179465, 0.334062, -0.022735, -0.129281, 0.124215, 0.207722, 0.191750, 0.191315, 0.045884, -0.090827, -0.066544, 0.002365, 0.109315, 0.257851, -0.100055, 0.056537, 0.025358, -0.000076, -0.209434, 0.356309, 0.200105, 0.191091, 0.067486, -0.024404, 0.130165, 0.040106, -0.103348, 0.118508, 0.207794, 0.016665, -0.172123, 0.131189, 0.128098, 0.056523, 0.214160, -0.104052, 0.057580, 0.124991, 0.002281, -0.061972, -0.115291, 0.152474, -0.024761, 0.133534, -0.101122, 0.052503, 0.191084, -0.174376, 0.013109, 0.000003, 0.045228, 0.061112, -0.000596, -0.093611, 0.001205, 0.341900, -0.047245, 0.134104, 0.033324, 0.231349, 0.000405, 0.041729, -0.000242, 0.001753, 0.005447, 0.015562, 0.227177, 0.001233, -0.220347, -0.116357, 0.000138, -0.000576, -0.001271, -0.024035, -0.078213, -0.004072, -0.056574, 0.007410, -0.030947, -0.140571, 0.136488, 0.000167, -0.019216, -0.004446, -0.214665, -0.000648, 0.124608, 0.234020, 0.119891, 0.196646, 0.331284, -0.290949, 0.254852, -0.030467, 0.253970, 0.249665, -0.096299, 0.000836, 0.044197, 0.175599, 0.024765, -0.063649, 0.323052, 0.134400, 0.078861, 0.160763, 0.100240, 0.161365, 0.012845, 0.000745, -0.053135, 0.048640, 0.087395, 0.000063, -0.061776, -0.117515, 0.000108, 0.001827, -0.219644, 0.016212, -0.102672, -0.022374, 0.035373, 0.116358, -0.000375, 0.257947, 0.054866, 0.000271, 0.169933, 0.000271, 0.103331, -0.046508, -0.000626, -0.077786, 0.195483, -0.008593, -0.000135, -0.029101, -0.000004, -0.320511, -0.166009, -0.059524, -0.100229, -0.139382, 0.127004, 0.022098, -0.000632, 0.311594, 0.406394, 0.000126, 0.182799, 0.072769, 0.064093, -0.150574, 0.038317, 0.117468, 0.000192, 0.074014, -0.173421, 0.034560, -0.006866, 0.000191, 0.016561, -0.213093, 0.040910, -0.059418, -0.255389, -0.024411, -0.024165, -0.059223, -0.000223, -0.004127, 0.011268, 0.091978, 0.029879, 0.154143, -0.207525, -0.006110, 0.201309, 0.014989, -0.056888, 0.000168, 0.041176, -0.010317, -0.000408, 0.231484, 0.112476, -0.272626, 0.039568, 0.221102, -0.182278, -0.017887, -0.058009, 0.000146, 0.021080, 0.022513, -0.207068, 0.054963, -0.009477, 0.056691, 0.156052, 0.376625, -0.194532, 0.132129, 0.142893, -0.004282, 0.043421, -0.067751, 0.114195, 0.114492, -0.000036, 0.038181, 0.147715, 0.039306, -0.119711, -0.122589, -0.172201, -0.047122, -0.070052, 0.233111, -0.004820, 0.012345, -0.006523, -0.043058, -0.007421, 0.016999, 0.239966, -0.014537, -0.008995, 0.282088, 0.003377, 0.188188, -0.054241, -0.191384, -0.000496, -0.218887, -0.178704, 0.101677, 0.036503, -0.001848, 0.106066, 0.000070, 0.038188, 0.040294, -0.003076, 0.055580, -0.033832, 0.136353, 0.171098, 0.000050, 0.032370, -0.073180, 0.022397, 0.122413, -0.028930, -0.000587, 0.051956, 0.023684, 0.188334, 0.260303, -0.003987, -0.128275, 0.127157, 0.052866, 0.057404, 0.192873, 0.097371, 0.118668, -0.019423, 0.076842, 0.030282, -0.123658, 0.138948, -0.000693, -0.064524, 0.026401, -0.028509, -0.017572, -0.000407, 0.017260, 0.025619, -0.000679, -0.121207, 0.156746, 0.000330, -0.088868, -0.077078, -0.022858, 0.126720, 0.000075, -0.079152, 0.099744, -0.001620, -0.032261, -0.011300, -0.185214, -0.005342, -0.089142, -0.000338, -0.169189, -0.070631, -0.128756, -0.164202, -0.135053, -0.002193, -0.000095, -0.000301, 0.144208, 0.091923, 0.005077, -0.081921, -0.250791, -0.000315, -0.102072, -0.152074, 0.314598, 0.003951, -0.027504, -0.000569, 0.064713, -0.025829, 0.138741, -0.000187, -0.191613, 0.161559, 0.020089, -0.083390, -0.095532, -0.054220, -0.089772, -0.059724, 0.075524, 0.003132, 0.009698, 0.019118, -0.120674, -0.175842, -0.000036, 0.079795, -0.005506, 0.279085, -0.322475, 0.018690, 0.139307, 0.186535, 0.218136, 0.019238, -0.024704, 0.127806, -0.073809, 0.132444, 0.156772, -0.003364, 0.000246, 0.352918, -0.104768, -0.130574, -0.014884, -0.071354, -0.194902, -0.026629, -0.015335, 0.000176, -0.076401, -0.098395, -0.000123, 0.070237, 0.098450, -0.020476, 0.000094, -0.062344, -0.034151, 0.050926, 0.000059, 0.006350, -0.015941, 0.118723, -0.036008, -0.079185, -0.057344, 0.108545, 0.006009, 0.096540, 0.047273, -0.143237, -0.077134, 0.187565, 0.056400, -0.105161, 0.164739, -0.022409, -0.223620, -0.094862, 0.010496, 0.000371, -0.000083, -0.000413, -0.181839, -0.076359, 0.077934, 0.118766, -0.000080, 0.159373, 0.338769, 0.096951, -0.063023, 0.169843, -0.016302, 0.174794, -0.071056, -0.042213, 0.027028, 0.108322, 0.000677, -0.153385, -0.022440, -0.168085, 0.000391, 0.243218, -0.085640, -0.051138, 0.000343, 0.128842, 0.000176, -0.121821, 0.085604, 0.061403, 0.195225, 0.128981, 0.000459, -0.168679, -0.004803, 0.175871, 0.121201, -0.083480, -0.062705, 0.075768, -0.000819, 0.086200, -0.072388, 0.191874, 0.224892, -0.041199, 0.131397, -0.041005, 0.116466, 0.228289, 0.210986, -0.117958, 0.000318, 0.256289, 0.277592, 0.077567, 0.239985, 0.320437, 0.004584, 0.105561, 0.065758, 0.182461, 0.202662, 0.062826, 0.000266, 0.025078, 0.000056, -0.074952, -0.000041, -0.000151, -0.000302, 0.002142, -0.000136, -0.029566, -0.040063, -0.068502, 0.120420, -0.000304, -0.161750, 0.127138, -0.000027, 0.022196, 0.071539, -0.192736, 0.000425, 0.117008, -0.239400, -0.000467, 0.057781, -0.252822, -0.197972, 0.047691, -0.014784, 0.080004, 0.189725, 0.083729, -0.000709, 0.005245, 0.015921, -0.198844, 0.106247, -0.000681, 0.203494, 0.200399, -0.072383, 0.000370, 0.214934, 0.056723, 0.097842, 0.000326, -0.265343, -0.013154, 0.026641, -0.016358, -0.176342, 0.088778, -0.004526, 0.194771, 0.109426, -0.135972, 0.017075, -0.044905, -0.020953, 0.040753, 0.077772, 0.000970, 0.003491, 0.112564, -0.011491, 0.240585, -0.056093, 0.098945, -0.000038, 0.224474, -0.036096, 0.002121, -0.030659, -0.372649, 0.049602, -0.386003, 0.037732, 0.010679, 0.007743, 0.122276, 0.220944, -0.182257, -0.174697, 0.211678, -0.001001, -0.124782, 0.039813, -0.029996, -0.101714, -0.018819, 0.048199, 0.181357, -0.235352, -0.234802, -0.461960, -0.114996, -0.048628, 0.025667, 0.159807, 0.122091, 0.000069, 0.062230, 0.006893, -0.002422, 0.030195, -0.191497, -0.175135, -0.000866, -0.078733, -0.111772, 0.170531, 0.000287, -0.015375, 0.181607, 0.000365, -0.001152, 0.086802, -0.043891, 0.115591, 0.188052, 0.229349, 0.196731, 0.094844, 0.166595, -0.114762, 0.000539, 0.068959, -0.000062, 0.165575};
#endif

#endif
