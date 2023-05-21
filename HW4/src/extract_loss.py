import re
import pdb

def extract_loss(log: str):
    loss_p = re.compile(r'GlobalStep: (\d+).*?loss: ([\d\.]+)')
    losses = loss_p.findall(log)
    for step, loss in losses:
        print(f'{step}\t\t{loss}')


if __name__ == '__main__':
    DATA_DIR = r'../resources'
    log_path  = f'{DATA_DIR}/384x24.log'
    with open(log_path, 'r', encoding='utf8') as fi:
        log = fi.read()

    extract_loss(log)