`ifndef __shift_reg_file_V__
`define __shift_reg_file_V__


module shift_reg_file
#(
    parameter DATA_WIDTH = 8,
    parameter VEC_LENGTH = 8
) (
    input  logic                   clk,
    input  logic                   reset,

    // @param d_in:  input data from on-chip SRAM that is written to the register file
    // @param w_en:  write enable signal. If True, then program the register file
    // @param r_en:  read enable signal. If True, then shift the register file by 1-bit
    // @param d_out: output data read from register file 
    input  logic [DATA_WIDTH-1:0]  d_in  [VEC_LENGTH-1:0], 
    input  logic                   r_en, 
    input  logic                   w_en,

    output logic                   d_out [VEC_LENGTH-1:0]
);
    genvar j;
    logic [DATA_WIDTH-1:0]  d_reg  [VEC_LENGTH-1:0];

    generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
            always @(posedge clk) begin
                if (reset) begin
                    d_reg[j] <= 0;
                end else if	(w_en) begin
                    d_reg[j] <= d_in[j];
                end else if	(r_en) begin
                    d_reg[j] <= (d_reg[j] << 1);
                end
            end
        end

        for (j=0; j<VEC_LENGTH; j=j+1) begin
            assign d_out[j] = d_reg[j][DATA_WIDTH-1];
        end
    endgenerate

endmodule

`endif