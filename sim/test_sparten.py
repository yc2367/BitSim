from sim.sparten import Sparten 
from model_profile.models.models import MODEL

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model architecture')
# imagenet dataset
parser.add_argument('--train_dir', type=str, default='./data/', help='training data directory')
parser.add_argument('--val_dir', type=str, default='./data/', help='test/validation data directory')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')

parser.add_argument('--workers', type=int, default=4,help='number of data loading workers (default: 4)')
parser.add_argument('--sparten_save_dir', type=str, default='./data/', help='sparten profiling save directory')

args = parser.parse_args()
name_list = ['resnet18', 'resnet50', 'mobilenet_v2']
name = args.model
model = MODEL[name]
model = model(weights='DEFAULT')

if __name__ == "__main__":
    acc = Sparten(8, 128, [32, 16], name, model, args)

    total_cycle    = acc.calc_cycle()
    compute_energy = acc.calc_compute_energy() / 1e6
    local_buffer_rd_energy = acc.calc_local_buffer_rd_energy() / 1e6
    local_buffer_wr_energy = acc.calc_local_buffer_wr_energy() / 1e6
    sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
    sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
    dram_energy    = acc.calc_dram_energy() / 1e6
    
    onchip_energy  = compute_energy + local_buffer_rd_energy + local_buffer_wr_energy + sram_rd_energy + sram_wr_energy
    total_energy   = (compute_energy + local_buffer_rd_energy + local_buffer_wr_energy + 
                      sram_rd_energy + sram_wr_energy + dram_energy)
    
    print(f'total cycle:               {total_cycle}')
    print(f'local buffer area:         {acc.local_buffer.area} mm2')
    print(f'weight buffer area:        {acc.w_sram.area} mm2')
    print(f'input buffer area:         {acc.i_sram.area} mm2')

    print(f'compute energy:            {compute_energy} uJ')
    print(f'local buffer read energy:  {local_buffer_rd_energy} uJ')
    print(f'local buffer write energy: {local_buffer_wr_energy} uJ')
    print(f'sram read energy:          {sram_rd_energy} uJ')
    print(f'sram write energy:         {sram_wr_energy} uJ')
    print(f'dram energy:               {dram_energy} uJ')
    print(f'on-chip energy:            {onchip_energy} uJ')
    print(f'total energy:              {total_energy} uJ')
    