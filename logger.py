import logging
import os
import time
class Logger:
    def __init__(self, log_file,write_log_level=False,name='default'):
        if os.path.exists(log_file):
            os.remove(log_file)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        # 创建文件处理程序
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        # 创建日志格式器
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        # 将文件处理程序添加到Logger对象中
        self.logger.addHandler(file_handler)

        self.logger.info('Log file created at %s' % time.asctime())


    def log(self, message):
        self.logger.info(message)

    def log_dic(self,config):
        self.logger.info("Config:")
        if isinstance(config, dict):
            for key,value in config.items():
                self.logger.info(f"\t--{key}:{value}")
        elif isinstance(config, object):
            for key in dir(config):
                if not key.startswith('_'):
                    value = getattr(config, key)
                    self.logger.info(f"\t--{key}:{value}")
        else:
            raise TypeError("Not supported type for config:%s"%type(config))
#test
if __name__ == '__main__':
    logger=Logger("train log.log")
    logger.log('This is a test message')
    logger.log(f"Epoch {6}: loss = {0.213}")

    dic={"a":1,"b":2}
    logger.log_dic(dic)