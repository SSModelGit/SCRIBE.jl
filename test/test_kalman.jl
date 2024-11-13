using Test
using SCRIBE

function main()
    return LGSFModelParameters()
end

@testset "Individual Agent Setup" begin
    @test typeof(main()) <: SCRIBEModelParameters
end