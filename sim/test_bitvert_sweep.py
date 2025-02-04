from sim.bitvert_sweep import BitVert 

name_list = [ 'resnet50', ]

if __name__ == "__main__":
    for name in name_list:
        for separator in [64, 128, 256, 512]:
            acc = BitVert(8, 8, 16, [32, 16], name, 
                        en_bbs=True, en_lsb_pruning=False, 
                        en_ol_channel=True, en_eager_compression=False, separator=separator)

            total_cycle    = acc.calc_cycle()
            compute_energy = acc.calc_compute_energy() / 1e6
            sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
            sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
            dram_energy    = acc.calc_dram_energy() / 1e6
            onchip_energy  = compute_energy + sram_rd_energy + sram_wr_energy
            total_energy   = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy
            
            print_energy = True
            print(f'model name: {name}')
            print(f'total cycle:        {total_cycle}')
            
            if print_energy:
                print(f'weight buffer area: {acc.w_sram.area} mm2')
                print(f'input buffer area:  {acc.i_sram.area} mm2')
                print(f'compute energy:     {compute_energy} uJ')
                print(f'sram rd energy:     {sram_rd_energy} uJ')
                print(f'sram wr energy:     {sram_wr_energy} uJ')
                print(f'dram energy:        {dram_energy} uJ')
                print(f'on-chip energy:     {onchip_energy} uJ')
                print(f'total energy:       {total_energy} uJ')

                acc.print_prec_eff()
            print()

