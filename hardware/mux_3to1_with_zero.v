`ifndef __mux_3to1_with_zero_V__
`define __mux_3to1_with_zero_V__

module mux_3to1_with_zero
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [1:0], 
    input  logic [1:0]             sel,     
    output logic [DATA_WIDTH-1:0]  out
);

    always_comb begin
        case (sel)
            2'b00:   out = vec[0];
            2'b01:   out = vec[1];
            2'b10:   out = 0;
            default: out = {DATA_WIDTH{1'bx}};
        endcase
    end

endmodule

`endif