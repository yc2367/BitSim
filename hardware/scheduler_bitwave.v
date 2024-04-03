`ifndef __scheduler_bitwave_V__
`define __scheduler_bitwave_V__

`include "reg_file.v"
`include "decoder_3to7.v"
`include "pencoder_7to3.v"
`include "mux_2to1.v"

module scheduler_bitwave
#(
    parameter DATA_WIDTH = 8,
    parameter SEL_WIDTH = $clog2(DATA_WIDTH)
) (
    input  logic                  clk,
    input  logic                  reset,
    input  logic                  en_comp,  // compute read-enable
    input  logic                  wen_rf,   // register file write-enable

    input  logic [DATA_WIDTH-1:0] col_mask , // indicate whether a bit-column is sparse or not

    output logic [SEL_WIDTH-1:0]  shift_ctr,
    output logic                  sign_ctrl,
    output logic [SEL_WIDTH-1:0]  nz_col_num
);
    logic [DATA_WIDTH-1:0]  din_rf;
    logic [DATA_WIDTH-1:0]  dout_rf;
    logic [DATA_WIDTH-2:0]  decoded_col_mask; 

    mux_2to1 #(DATA_WIDTH-1) mux (
        .in_1 (col_mask[DATA_WIDTH-2:0]), 
        .in_0 (decoded_col_mask),
        .sel  (wen_rf),
        .out  (din_rf[DATA_WIDTH-2:0])
    );
    assign din_rf[DATA_WIDTH-1] = col_mask[DATA_WIDTH-1];

    reg_file #(
        .DATA_WIDTH(DATA_WIDTH), .VEC_LENGTH(1)
    ) rf (
        .d_in(din_rf), .d_out(dout_rf), .w_en(wen_rf), .*
    );

    logic [SEL_WIDTH-1:0]   shift_ctr_tmp;
    logic                   val_shift_ctr_tmp;
    logic [DATA_WIDTH-2:0]  dec_out;
    pencoder_7to3  pen (
        .bitmask(dout_rf[DATA_WIDTH-2:0]), 
        .out(shift_ctr_tmp), .val(val_shift_ctr_tmp)
    );
    decoder_3to7   dec (
        .in(shift_ctr_tmp), .val(val_shift_ctr_tmp), .out(dec_out)
    );
    assign decoded_col_mask = dout_rf[DATA_WIDTH-2:0] ^ dec_out;

    logic [SEL_WIDTH-1:0] nz_col_num_tmp;
    always_comb begin
        nz_col_num_tmp = '0;  
        foreach(col_mask[idx]) begin
            nz_col_num_tmp += col_mask[idx];
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            shift_ctr   <= 0;
            sign_ctrl   <= 0;
            nz_col_num  <= 0;
        end else if	(en_comp) begin
            shift_ctr   <= shift_ctr_tmp;
            sign_ctrl   <= dout_rf[DATA_WIDTH-1];
            nz_col_num  <= nz_col_num_tmp;
        end
    end
endmodule

`endif