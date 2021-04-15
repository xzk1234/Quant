from core.base import BaseTask  # 所有任务类都需要继承这个类
from apscheduler.triggers.cron import CronTrigger  # 直接使用APScheduler中的触发器设计
from sqlalchemy import create_engine
from main_task import *

parser = argparse.ArgumentParser(description='Quant')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--display_step', type=int, default=100)
parser.add_argument('--n_steps_encoder', type=int, default=10)
parser.add_argument('--n_hidden_encoder', type=int, default=128)
parser.add_argument('--n_input_decoder', type=int, default=1)
parser.add_argument('--n_steps_decoder', type=int, default=10)
parser.add_argument('--n_hidden_decoder', type=int, default=128)
parser.add_argument('--n_classes', type=int, default=2)
args = parser.parse_args()

Tend_model = TendencyPredict(args, sql_price_name='price_new', sql_return_name='return_new', sql_predict_name='predict_new')

class UpdateStockDataDaily(BaseTask):
    def __init__(self):
        super().__init__(
            task_name='read_data_fromJQ',
            schema_name="basic_data",  # 保存数据的数据库schema名称
            table_name="JQ_HS300_price",  # 保存数据的数据表名称
            trigger=CronTrigger(minute='*'))
        print("init UpdateStockDaily")
        # 如果定义了schema_name和table_name，save阶段会相应入库，不需要单独实现

    def gather(self):
        last_price = Tend_model.inference()
        self.gathered_data = last_price  # 存储到中间变量中
        return True  # 注意只有返回值为True才会继续进行下一阶段的任务

    def apply(self):
        # 将从万得中取得的数据进行调整
        print('call apply in UpdateStockDataDaily')
        columns = self.gathered_data.columns
        new_cols = {}
        for c in columns:
            new_cols[c] = c.lower()
        self.gathered_data.rename(columns=new_cols, inplace=True)
        self.processed_data = self.gathered_data
        return True  # 注意只有返回值为True才会继续进行下一阶段的任务

task = UpdateStockDataDaily()