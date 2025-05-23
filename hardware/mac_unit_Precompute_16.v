`ifndef __mac_unit_Precompute_16_V__
`define __mac_unit_Precompute_16_V__


module mux_3to1
#(
    parameter DATA_WIDTH = 8
) (
    input  logic signed [DATA_WIDTH-1:0]  in_1, 
    input  logic signed [DATA_WIDTH-1:0]  in_2, 
    input  logic signed [DATA_WIDTH:0]    sum, 
    input  logic        [1:0]             sel,   
    output logic signed [DATA_WIDTH:0]    out
);
	localparam OUT_WIDTH = DATA_WIDTH + 1;
    always_comb begin
        case (sel) 
            2'b00:   out = 0;
            2'b01:   out = in_1;
            2'b10:   out = in_2;
            2'b11:   out = sum;
            default: out = {OUT_WIDTH{1'bx}};
        endcase
    end
endmodule


module pos_neg_select #(
	parameter DATA_WIDTH = 8
) (
	input  logic signed [DATA_WIDTH-1:0] in,
	input  logic                         sign,	
	output logic signed [DATA_WIDTH-1:0] out
); 
	always_comb begin
		if (sign) begin
			out = ~in + 1'b1;
		end else begin
			out = in;
		end
	end
endmodule


module shifter_3bit #(
	parameter IN_WIDTH  = 12,
	parameter OUT_WIDTH = 19
) (
	input  logic signed [IN_WIDTH-1:0]  in,
	input  logic        [2:0]           shift_sel,	
	output logic signed [OUT_WIDTH-1:0] out
);
	always_comb begin 
		case (shift_sel)
			3'b000 : out = in;
			3'b001 : out = in <<< 1;
			3'b010 : out = in <<< 2;
			3'b011 : out = in <<< 3;
			3'b100 : out = in <<< 4;
			3'b101 : out = in <<< 5;
			3'b110 : out = in <<< 6;
			3'b111 : out = in <<< 7;
			default: out = {OUT_WIDTH{1'bx}};
		endcase
	end
endmodule


module mac_unit_Precompute_16
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 16,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act_in      [VEC_LENGTH-1:0], 
	input  logic signed [DATA_WIDTH:0]       act_in_psum [VEC_LENGTH/2-1:0], 
	input  logic                             w_bit       [VEC_LENGTH-1:0],

	input  logic        [2:0]                column_idx,    // current column index for shifting 
	input  logic                             is_msb,
	input  logic signed [ACC_WIDTH-1:0]      accum_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;

	logic signed [DATA_WIDTH:0]  adder_in  [VEC_LENGTH/2-1:0] ;
	generate
		for (j=0; j<VEC_LENGTH/2; j=j+1) begin
			mux_3to1 #(DATA_WIDTH) value_select (
				.in_1(act_in[2*j]), .in_2(act_in[2*j+1]), .sum(act_in_psum[j]), 
				.sel(w_bit[(2*j+1):(2*j)]), .out(adder_in[j])
			);
		end
	endgenerate

	logic signed [DATA_WIDTH+1:0]  psum_1 [VEC_LENGTH/4-1:0];
	logic signed [DATA_WIDTH+2:0]  psum_2 [VEC_LENGTH/8-1:0];
	logic signed [DATA_WIDTH+3:0]  psum_total;
	generate
		for (j=0; j<VEC_LENGTH/4; j=j+1) begin
			assign psum_1[j] = adder_in[2*j] + adder_in[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/8; j=j+1) begin
			assign psum_2[j] = psum_1[2*j] + psum_1[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/16; j=j+1) begin
			assign psum_total = psum_2[2*j] + psum_2[2*j+1];
		end
	endgenerate

	logic signed [DATA_WIDTH+3:0]   psum_act_shift_in;
	logic signed [DATA_WIDTH+10:0]  psum_act_shift_out, psum_act_shift_reg;;
	pos_neg_select #(DATA_WIDTH+4) twos_complement (.in(psum_total), .sign(is_msb), .out(psum_act_shift_in));
	shifter_3bit #(.IN_WIDTH(DATA_WIDTH+4), .OUT_WIDTH(DATA_WIDTH+11)) shift_psum (
		.in(psum_act_shift_in), .shift_sel(column_idx), .out(psum_act_shift_out)
	);

	logic signed [ACC_WIDTH-1:0]  accum_in, accum_out;
	always_comb begin
		if (load_accum) begin
			accum_in = accum_prev;
		end else begin
			accum_in = accum_out;
		end
	end

	always @(posedge clk) begin
		if (reset) begin
			accum_out <= 0;
			psum_act_shift_reg <= 0;
		end else if	(en) begin
			psum_act_shift_reg <= psum_act_shift_out;
			accum_out <= psum_act_shift_reg + accum_in;
		end
	end

	assign result = accum_out[ACC_WIDTH-1:ACC_WIDTH-16];

endmodule


module mac_unit_Precompute_16_clk
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 16,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act         [VEC_LENGTH-1:0], 
	input  logic signed [DATA_WIDTH:0]       act_in_psum [VEC_LENGTH/2-1:0], 
	input  logic                             w_bit       [VEC_LENGTH-1:0],

	input  logic        [2:0]                column_idx,    // current column index for shifting 
	input  logic                             is_msb,
	input  logic signed [ACC_WIDTH-1:0]      accum_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;
	
	logic signed [DATA_WIDTH-1:0]  act_in [VEC_LENGTH-1:0];
	generate
	for (j=0; j<VEC_LENGTH; j=j+1) begin
		always @(posedge clk) begin
			if (reset) begin
				act_in[j] <= 0;
			end else begin
				act_in[j] <= act[j];
			end
		end
	end
	endgenerate

	mac_unit_Precompute_16 #(DATA_WIDTH, VEC_LENGTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif