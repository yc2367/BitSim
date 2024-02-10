`ifndef __mac_unit_16_signed_V__
`define __mac_unit_16_signed_V__


module pos_neg_select #(
	parameter DATA_WIDTH = 8
) (
	input  logic signed [DATA_WIDTH-1:0] in,
	input  logic                         sign,	
	output logic signed [DATA_WIDTH:0]   out
); 
	logic signed [DATA_WIDTH:0] in_extended;
	assign in_extended = {in[DATA_WIDTH-1], in};

	always_comb begin
		if (sign) begin
			out = ~in_extended + 1'b1;
		end else begin
			out = in_extended;
		end
	end
endmodule


module value_select #(
	parameter DATA_WIDTH = 9
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


module mac_unit_16_signed
#(
    parameter DATA_WIDTH = 8,
	parameter VEC_LENGTH = 16
) (
	input  logic                          clk,
	input  logic                          reset,
	input  logic signed [DATA_WIDTH-1:0]  act   [VEC_LENGTH-1:0], 
	input  logic                          sign  [VEC_LENGTH-1:0],
	input  logic                          w_bit [VEC_LENGTH-1:0],
	input  logic        [2:0]             column_idx,
	output logic signed [DATA_WIDTH+16:0] result
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

	logic signed [DATA_WIDTH:0]  act_out    [VEC_LENGTH-1:0] ;
	logic signed [DATA_WIDTH:0]  bs_mul_out [VEC_LENGTH-1:0] ;
	generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
			pos_neg_select #(DATA_WIDTH) select_1 (
				.in(act_in[j]), .sign(sign[j]), .out(act_out[j])
			);
			value_select #(DATA_WIDTH+1) select_2 (
				.in(act_out[j]), .w_bit(w_bit[j]), .out(bs_mul_out[j])
			);
		end
	endgenerate

	logic signed [DATA_WIDTH+1:0]  psum_1 [VEC_LENGTH/2-1:0];
	logic signed [DATA_WIDTH+2:0]  psum_2 [VEC_LENGTH/4-1:0];
	logic signed [DATA_WIDTH+3:0]  psum_3 [VEC_LENGTH/8-1:0];
	logic signed [DATA_WIDTH+3:0]  psum_4 ;

	generate
		for (j=0; j<VEC_LENGTH/2; j=j+1) begin
			assign psum_1[j] = bs_mul_out[2*j] + bs_mul_out[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/4; j=j+1) begin
			assign psum_2[j] = psum_1[2*j] + psum_1[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/8; j=j+1) begin
			assign psum_3[j] = psum_2[2*j] + psum_2[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/16; j=j+1) begin
			assign psum_4 = psum_3[2*j] + psum_3[2*j+1];
		end
	endgenerate

	logic signed [DATA_WIDTH+10:0]  shifted_psum;
	shifter #(.IN_WIDTH(DATA_WIDTH+4), .OUT_WIDTH(DATA_WIDTH+11)) shift (
		.in(psum_4), .shift_sel(column_idx), .out(shifted_psum)
	);

	always @(posedge clk) begin
		if (reset) begin
			result <= 0;
		end else begin
			result <= result + shifted_psum;
		end
	end

endmodule

`endif