import logging
import math

from models.modules.Inv_arch import *
from models.modules.Subnet_constructor import subnet

logger = logging.getLogger('base')

####################
# define network
####################
def define_G_v2(opt):
	opt_net = opt['network_G']
	which_model = opt_net['which_model_G']
	subnet_type = which_model['subnet_type']
	opt_datasets = opt['datasets'] #train, test에 대한 정보가 많이 들어있음.
	down_num = int(math.log(opt_net['scale'], 2))
	if opt['num_image'] == 1:
		netG = VSN(opt, subnet(subnet_type, 'xavier'), subnet(subnet_type, 'xavier'), down_num)
	else:
		netG = VSN(opt, subnet(subnet_type, 'xavier'), subnet(subnet_type, 'xavier_v2'), down_num)

	return netG
