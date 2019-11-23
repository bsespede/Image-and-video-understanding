from stvi_class.datapreprocessing import DataProcessor
from stvi_class.featureextraction import FeatureExtractor

dps = DataProcessor('../../demoData/')
fet = FeatureExtractor(dps)
dps.processVideo('twist0.pkl')
# dps.showSTVI(100)
fet.processSTVIs()
