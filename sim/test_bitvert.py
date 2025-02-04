from sim.bitvert import BitVert 
from model_profile.models.models import MODEL

name_list = ['vgg16', 'resnet34', 'resnet50', 'vit-small', 'vit-base', 'bert-mrpc', 'bert-sst2']

if __name__ == "__main__":
    for name in name_list:
        en_per_layer = False # whether enable per-layer search
        if en_per_layer: 
            if name in ['vit-small', 'bert-sst2']:
                w_prec = 7
            elif name in ['resnet34', 'resnet50']:
                w_prec = 7.2
            elif name in ['vgg16']:
                w_prec = 7.2
            else:
                w_prec = 6.5
        else:
            w_prec = 8
            
        acc = BitVert(w_prec, 8, 16, [32, 16], name, 
                    en_bbs=True, en_lsb_pruning=False, 
                    en_ol_channel=True, en_eager_compression=True)

        total_cycle    = acc.calc_cycle()
        compute_energy = acc.calc_compute_energy() / 1e6
        sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
        sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
        dram_energy    = acc.calc_dram_energy() / 1e6
        onchip_energy  = compute_energy + sram_rd_energy + sram_wr_energy
        total_energy   = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy
        
        print_energy = False
        print(f'model name:     {name}')
        acc.print_prec_eff()
        print(f'total cycle:    {total_cycle}')
        
        if print_energy:
            print(f'weight buffer area: {acc.w_sram.area} mm2')
            print(f'input buffer area:  {acc.i_sram.area} mm2')
            print(f'compute energy:     {compute_energy} uJ')
            print(f'sram rd energy:     {sram_rd_energy} uJ')
            print(f'sram wr energy:     {sram_wr_energy} uJ')
            print(f'dram energy:        {dram_energy} uJ')
            print(f'on-chip energy:     {onchip_energy} uJ')
            print(f'total energy:       {total_energy} uJ')

        print()

