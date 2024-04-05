from sim.bitwave import Bitwave
from model_profile.models.models import MODEL

name_list = ['resnet18', 'resnet50', 'mobilenet_v2']

if __name__ == "__main__":
    for i in range(3):
        name = name_list[i]
        model = MODEL[name]
        model = model()
        acc = Bitwave(8, 8, 8, [32, 16], name, model, layer_prec={}, en_bitflip=True)
        
        total_cycle    = acc.calc_cycle()
        compute_energy = acc.calc_compute_energy() / 1e6
        sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
        sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
        dram_energy    = acc.calc_dram_energy() / 1e6
        onchip_energy  = compute_energy + sram_rd_energy + sram_wr_energy
        total_energy   = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy
        
        print(f'total cycle:        {total_cycle}')
        print(f'weight buffer area: {acc.w_sram.area} mm2')
        print(f'input buffer area:  {acc.i_sram.area} mm2')
        print(f'compute energy:     {compute_energy} uJ')
        print(f'sram rd energy:     {sram_rd_energy} uJ')
        print(f'sram wr energy:     {sram_wr_energy} uJ')
        print(f'dram energy:        {dram_energy} uJ')
        print(f'on-chip energy:     {onchip_energy} uJ')
        print(f'total energy:       {total_energy} uJ')
        
        print()
        