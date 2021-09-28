# using Random, Colors, Statistics, DelimitedFiles, Arpack, Distributions, Images, CSV, LinearAlgebra, SparseArrays, PlotlyJS, ImageFiltering
using DelimitedFiles, Images

function maxPooling(a)
  AConv=fill(0.0,(Int(size(a,1)),Int(size(a,2))))
  for i=2:size(a,1)-1, j=2:size(a,2)-1
    if a[i,j]==1
        AConv[i,j]=1
    else
        obj=sum([a[i+k,j+l] for k=-1:1, l=-1:1]);
        obj > 6 ? AConv[i,j]=1 : 1
    end
  end
  return AConv
end

function poolIt(a)
    aa=maxPooling(a)
    for i=1:10
        aa=maxPooling(aa)
    end
    return aa
end

function readAndDir(readDir, writeDir)
  list=readdir(readDir)
  if isdir(writeDir) == false
     mkdir(writeDir)
   end
   return list
end

function convertToBW(img)
    bitAr = length(size(img)) == 3 ? (sum(img, dims=1) .> 0.0)[1,:,:] : img .> 0.0
    return bitAr
end

function skeletonize(readDir, writeDir)
    @info "Read from $readDir "
    @info "Write to $writeDir"
    list = readAndDir(readDir, writeDir)

    for i=1:size(list,1)
        @info i
        l=readDir * list[i]
        @info "Loading $l"
        img = channelview((load(l)))
        img=convertToBW(img)
        @info "Pooling $l"
        # a=poolIt(aa[1,:,:])
        a=poolIt(img)
        save(l, a)
        @info "Skeletonizing $l"
        run(`python3 sket.py $l`)
        println(l)
        newl=list[i][1:end-4]*"_Sket.tif"
        run(`mv $newl $writeDir `)
        run(`cp $l $writeDir `)
        @warn "$writeDir"
    end
end


cd("/Users/javier/Desktop/SyntheticVasculature/extractNetwork")
PATH_IN = "/Users/javier/Desktop/SyntheticVasculature/Data/DRIVE/"
PATH_IN = "/Users/javier/Desktop/SyntheticVasculature/Data/HRF_AV_GT/"
PATH_OUT = "/Users/javier/Desktop/SyntheticVasculature/Data/sket/"

skeletonize(PATH_IN, PATH_OUT)

# readAndDir(PATH_IN, PATH_OUT)


# files = readdir(PATH_IN)
# for file in files
#     if split(file,".")[2] != "gif"
#         img  = load(PATH_IN * "/" * file)
#         save(PATH_IN * "/" * file[1:end-3] * "gif", img )
#         rm(PATH_IN * "/" * file)
#     end
# end
