using Random, Colors, Statistics, DelimitedFiles, Arpack, Distributions, Images, CSV, LinearAlgebra, SparseArrays, PlotlyJS, ImageFiltering

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
  list=readdir("./" * string(readDir) * "/")
  if (isdir(string(writeDir))) == false
     mkdir(string(writeDir))
   end
   return list
end

function main(readDir, writeDir)
    println("Read from = " * string(readDir))
    println("Write to = " * string(writeDir))
    list = readAndDir(readDir, writeDir)

    for i=7:size(list,1)
        l="./" * string(readDir) * "/" * list[i]
        println("arc = "*string(l))
        aa=channelview((load(l)))
        a=poolIt(aa[1,:,:])
        save(l, a)
        run(`python3 sket.py $l`)
        println(l)
        newl=l[8:length(l)-4]*"_Sket.tif"
        run(`mv $newl $writeDir `)
        run(`cp $l $writeDir `)
    end
end



main("toSK", "sket")

`python3 `
