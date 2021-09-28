using Plots, DelimitedFiles, CSV, DataFrames

mutable struct netAbs
    nodes
    adj
    ep
end

function loadData(;dir="networks")
    files = readdir(pwd() * "/$(dir)/")
    nodesFiles = files[findall(x->endswith(x,"nodes.csv"), files)]
    adjFiles = files[findall(x->endswith(x,"adj.csv"), files)]
    epFiles = files[findall(x->endswith(x,"endpoints.csv"), files)]
    net = Array{netAbs}(undef, size(nodesFiles,1))
    if size(nodesFiles,1) == size(adjFiles,1) == size(epFiles,1)
        for (i,node) in enumerate(nodesFiles)
            # @info i, node, adj, ep
            loadedNodes = Array(CSV.read(pwd() * "/networks/$(node)", DataFrame, header=0 ))[:,1:2]
            loadedAdj = Array(CSV.read(pwd() * "/networks/$(split(node, "_nodes")[1] * "_adj.csv")", DataFrame, header=0 ))
            loadedEndPoints = Array(CSV.read(pwd() * "/networks/$(split(node, "_nodes")[1] * "_endpoints.csv")", DataFrame, header=0 ))[:,1]
            net[i] = netAbs(loadedNodes, loadedAdj, loadedEndPoints)
        end
    else
        @warn "Not same number of nodes, adj and endpoint files."
    end
    return net
end

function plotNetwork(idx)
    fig = Plots.hline(0)
    for i=1:size(net[idx].nodes,1), j=1:size(net[idx].nodes,1)
        if net[idx].adj[j,i]==1
            fig = plot!([net[idx].nodes[j,2],net[idx].nodes[i,2]],[net[idx].nodes[j,1],net[idx].nodes[i,1]],
              markershapes = :circle, lw=2, ms=5, legend=:none, c=:black, markerstrokewidth=0)
        end
    end
    fig = Plots.hline!(0)
    return fig
end

cd("/Users/javier/Desktop/SyntheticVasculature/Data")
net = loadData()

using Interact

@manipulate for i in 1:size(net,1)
    f1 = plot(reshape(sum(net[1].adj, dims=1),:), seriestype=:barhist, width=0, normalize=true, tickfont = font(10, "Helvetica"), frame=:box, size=(700,700), legend=false, xlabel="Node degree", ylabel="PDF")
    f2 = plotNetwork(i)
    plot(f1,f2, size=(2500,2000), layout = (2, 1))
end

# plot(reshape(sum(net[1].adj, dims=1),:), seriestype=:barhist, width=0, normalize=true, tickfont = font(10, "Helvetica"), frame=:box, size=(700,700), legend=false, xlabel="Node degree", ylabel="PDF")
