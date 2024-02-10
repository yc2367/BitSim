`ifndef __mux_5to1_with_zero_V__
`define __mux_5to1_with_zero_V__

module mux_5to1_with_zero
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [3:0], 
    input  logic [2:0]             sel,     
    output logic [DATA_WIDTH-1:0]  out
);

    always_comb begin
        case (sel) // synopsys infer_mux
            3'b000:  out = vec[0];
            3'b001:  out = vec[1];
            3'b010:  out = vec[2];
            3'b011:  out = vec[3];
            3'b100:  out = 0;
            default: out = {DATA_WIDTH{1'bx}};
        endcase
    end

endmodule

`endif