import configparser

class ConfigReader():
    def __init__(self, argv):
        self.arguments = argv
        confFilePath = '../conf/Sentinel2.conf'
        if len(self.arguments) > 2:
            confFilePath = self.arguments[2]
        print(f'Reading {confFilePath} ......')
        self.config = configparser.ConfigParser()
        self.config.read(confFilePath, encoding='utf-8')
        self.add_exec_tag = ''
        if len(self.arguments) > 3:
            self.add_exec_tag = self.arguments[3]


    def getConfFromFile(confFilePath: str):
        config = configparser.ConfigParser()
        return config.read(confFilePath, encoding='utf-8')


    def getConf(self):
        return self.config

    def getFromConf(self, section: str, name: str):
        config = self.getConf()
        return config[section].get(name)

    def getExecutionTag(self):
        config = self.getConf()
        dataset = self.arguments[1]
        options = config['options']
        settings = config['settings']

        tag = f'_{settings.get("size")}'
        tag += f'_{settings.get("optimize_by")}'

        if self.add_exec_tag:
            tag += f'_{self.add_exec_tag}'
        # tag += f'_fourth-try'
        if int(options.get('addIndexAndNormalize')) == 1:
            tag += '_scl_veg_norm'

        if int(settings.get('sampled')) == 1:
            tag += f'_sampled{settings.get("sampling_size")}'

        if int(settings.get('loss_weight')) == 1:
            tag += f'_weighted'

        if int(settings.get('setChannelAttentionLayer')) == 1:
            tag += '_ch-att'

        if int(settings.get('setAttentionLayer')) == 1:
            tag += '_att-layer'

        tag += f'_t{settings.get("trials")}'
        tag += f'_e{settings.get("epoches")}'
        tag += f'_{dataset}'
        return tag