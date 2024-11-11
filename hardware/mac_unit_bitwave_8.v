`ifndef __mac_unit_Wave_8_V__
`define __mac_unit_Wave_8_V__


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


module value_select #(
	parameter DATA_WIDTH = 8
) (
	input  logic signed [DATA_WIDTH-1:0] in,
	input  logic                         w_bit,	
	output logic signed [DATA_WIDTH-1:0] out
); 
	always_comb begin
		if (w_bit) begin
			out = in;
		end else begin
			out = 0;
		end
	end
endmodule


module shifter #(
	parameter IN_WIDTH  = 11,
	parameter OUT_WIDTH = IN_WIDTH + 6
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
			default: out = {OUT_WIDTH{1'bx}};
		endcase
	end
endmodule


module mac_unit_Wave_8
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act_in   [VEC_LENGTH-1:0], 
	input  logic                             sign     [VEC_LENGTH-1:0],
	input  logic                             w_bit    [VEC_LENGTH-1:0],
	input  logic        [2:0]                column_idx,
	input  logic signed [ACC_WIDTH-1:0]      accum_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;

	logic signed [DATA_WIDTH-1:0]  act_out    [VEC_LENGTH-1:0] ;
	logic signed [DATA_WIDTH-1:0]  bs_mul_out [VEC_LENGTH-1:0] ;
	generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
			pos_neg_select #(DATA_WIDTH) select_1 (
				.in(act_in[j]), .sign(sign[j]), .out(act_out[j])
			);
			value_select #(DATA_WIDTH) select_2 (
				.in(act_out[j]), .w_bit(w_bit[j]), .out(bs_mul_out[j])
			);
		end
	endgenerate

	logic signed [DATA_WIDTH+0:0]  psum_1 [VEC_LENGTH/2-1:0];
	logic signed [DATA_WIDTH+1:0]  psum_2 [VEC_LENGTH/4-1:0];
	logic signed [DATA_WIDTH+2:0]  psum_total;
	generate
		for (j=0; j<VEC_LENGTH/2; j=j+1) begin
			assign psum_1[j] = bs_mul_out[2*j] + bs_mul_out[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/4; j=j+1) begin
			assign psum_2[j] = psum_1[2*j] + psum_1[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/8; j=j+1) begin
			assign psum_total = psum_2[2*j] + psum_2[2*j+1];
		end
	endgenerate

	logic signed [DATA_WIDTH+8:0]  shifted_psum, shifted_psum_reg;
	shifter #(.IN_WIDTH(DATA_WIDTH+3), .OUT_WIDTH(DATA_WIDTH+9)) shift (
		.in(psum_total), .shift_sel(column_idx), .out(shifted_psum)
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
			shifted_psum_reg <= 0;
		end else if	(en) begin
			shifted_psum_reg <= shifted_psum;
			accum_out <= shifted_psum_reg + accum_in;
		end
	end

	assign result = accum_out[ACC_WIDTH-1:ACC_WIDTH-16];

endmodule


module mac_unit_Wave_8_clk
#(
	parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act      [VEC_LENGTH-1:0], 
	input  logic                             sign_in  [VEC_LENGTH-1:0],
	input  logic                             w_b      [VEC_LENGTH-1:0],
	input  logic        [2:0]                column_idx_in,
	input  logic signed [ACC_WIDTH-1:0]      accum_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;
	
	logic signed [DATA_WIDTH-1:0]  act_in [VEC_LENGTH-1:0];
	logic                          sign   [VEC_LENGTH-1:0];
	logic                          w_bit  [VEC_LENGTH-1:0];
	logic        [2:0]             column_idx;
	generate
	for (j=0; j<VEC_LENGTH; j=j+1) begin
		always @(posedge clk) begin
			if (reset) begin
				act_in[j] <= 0;
				sign[j] <= 0;
				w_bit[j] <= 0;
			end else begin
				act_in[j] <= act[j];
				sign[j]   <= sign_in[j];
				w_bit[j]  <= w_b[j];
			end
		end
	end
	endgenerate

	always @(posedge clk) begin
		if (reset) begin
			column_idx <= 0;
		end else begin
			column_idx <= column_idx_in;
		end
	end

	mac_unit_Wave_8 #(DATA_WIDTH, VEC_LENGTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif