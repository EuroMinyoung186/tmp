import logging
logger = logging.getLogger('base')

def create_model(opt):
    model = opt['model'] #MIMO-VRN-h
    frame_num = opt['gop'] #1 (frame_num)
    from .IBSN import Model_VSN as M

    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m