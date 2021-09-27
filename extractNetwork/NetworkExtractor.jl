# using Random, Statistics, DelimitedFiles, Arpack, Distributions
# using Images, CSV, LinearAlgebra, SparseArrays, Plots

using Plots, Images, DelimitedFiles, Statistics

begin
  function init2(arg)            #THIS FUNCTION IMPORTS THE SKELETONIZED PIC AND
                              #RETURNS THE GRAYSCALE PIC
    if arg==0
      estrin="/Users/javier/Desktop/Retinopathy/DRIVE/test/1st_manual/01_manual1_Sket.tif"
    else
      estrin=string(arg)
    end
    pic=load(estrin)
    GPic=Gray.(pic)
    return GPic
  end

  function addZeros(a)
    A=channelview(a)
    if size(A)[1]- size(A)[2]>0
      A=hcat(A,fill(0.0, (size(A)[1], size(A)[1]- size(A)[2]) ))
    elseif size(A)[1]- size(A)[2]<0
      A=vcat(A,fill(0.0, ( -size(A)[1]+ size(A)[2], size(A)[2]) ))
    end
    return A
  end

  function checkIfNeighborIsNode(nod,x,y)
    arj= [x y].==nod
    r = arj[:,1]'*arj[:,2]

    if r==1
      return true
    elseif r==0
      return false
    else
      println("Your code is doing some nasty things...")
      return true
    end
  end
  println(pwd())
  function removeOverlapNodes(a)
    newA=copy(a)              #SKELETONIZED WITHOUT NODES (THIS MATRIX IS BINARY)
    nod=[0 0]               #POSITION OF NODES AND ENDPOINTS
    numberOfSegPerNode=[0]    #NUMBER OF LINKS FROM GIVEN NODE (COUPLED WITH PREVIOUS ARRAY)
    adjMat=zeros(size(a)[1],size(a)[2])       #MISLEADING NAME. BINARY MATRIX WHICH CONTAINS ONLY THE NODES
    endpoints=zeros(size(a)[1],size(a)[2])    #BINARY MATRIX WHICH CONTAINS ONLY THE ENDPOINTS
    ep=[0 0]          #POSITIONS OF ENDPOINTS

    for i=1:size(a)[1], j=1:size(a)[2]
      counter=0;
      isitaNeighbor=false
      if a[i,j]==1.0
        if a[i-1,j]==1.0
          counter +=1
          isitaNeighbor=checkIfNeighborIsNode(nod,i-1,j)
        end
        if a[i+1,j]==1.0
          counter +=1
          isitaNeighbor = isitaNeighbor ? true : checkIfNeighborIsNode(nod,i+1,j)
        end
        if a[i,j-1]==1.0
          counter +=1
          isitaNeighbor = isitaNeighbor ? true : checkIfNeighborIsNode(nod,i,j-1)
        end
        if a[i,j+1]==1.0
          counter +=1
          isitaNeighbor = isitaNeighbor ? true : checkIfNeighborIsNode(nod,i,j+1)
        end
        if a[i+1,j+1]==1.0
          counter +=1
          isitaNeighbor = isitaNeighbor ? true : checkIfNeighborIsNode(nod,i+1,j+1)
        end
        if a[i-1,j+1]==1.0
          counter +=1
          isitaNeighbor = isitaNeighbor ? true : checkIfNeighborIsNode(nod,i-1,j+1)
        end
        if a[i+1,j-1]==1.0
          counter +=1
          isitaNeighbor = isitaNeighbor ? true : checkIfNeighborIsNode(nod,i+1,j-1)
        end
        if a[i-1,j-1]==1.0
          counter +=1
          isitaNeighbor = isitaNeighbor ? true : checkIfNeighborIsNode(nod,i-1,j-1)
        end
      end

      if isitaNeighbor
        #println("It is a neighbor node!")
      else
        if counter>2 || counter==1
          #println("removing pixel")
          for k=-1:1, l=-1:1
            newA[i+k,j+l]=0.0
          end
          adjMat[i,j]=1.0
          nod=vcat(nod,[i j])
          append!(numberOfSegPerNode,counter)
          if counter==1
            #adjMat[i,j]=1.0
            endpoints[i,j]=1.0
            newA[i,j]=0.0
            ep=vcat(ep,[i j])
            #println("1 neighbor")
          end
        end
      end
    end
    return newA, nod[2:size(nod)[1],:], adjMat, ep[2:size(ep,1),:], numberOfSegPerNode[2:size(numberOfSegPerNode)[1]]
  end

  function algSearch(a, nod, index, newNode, ASeg)
    initialX, initialY=nod[newNode,1], nod[newNode,2]
    oldvalues=[initialX initialY]
    initialnod=[nod[newNode,1] nod[newNode,2]]
    firstneighborlist=[0 0]
    outputNodes=[0 0]
    diamMean=[0.0]
    traj=[initialX initialY]
    qq=false
    for ind=1:index
      diamArray=[0]

      for tim=1:2000
          if tim>1
              newx, newy, qq = secondNeiIsNode(nod, initialX, initialY, oldvalues[1], oldvalues[2],initialnod)
              if qq
                initialX=newx
                initialY=newy
                traj=vcat(traj,[initialX initialY])
                #println("if qq " * string([initialX initialY]) * "  " * string([nod[newNode,1] nod[newNode,2]]))
                break
              end
          end

          for k=-1:1, l=-1:1
            if [k,l] !=[0,0] && a[initialX+k, initialY+l]==1 && [initialX+k initialY+l]!=oldvalues && [initialX+k initialY+l]!=initialnod && NotInthelis([initialX+k initialY+l], firstneighborlist)
              #if ind==index
                oldvalues=[initialX initialY]
                initialX=initialX+k
                initialY=initialY+l
                traj=vcat(traj,[initialX initialY])

                if tim<=100 #tim>1 && tim%2==0 &&
                  diam=measureDiam(ASeg, oldvalues[1], oldvalues[2], initialX, initialY)
                  if diam <=10
                    #global diamMat[oldvalues[1]-Int(floor(diam/2)*(oldvalues[2]-initialY)): oldvalues[1]+ Int(floor(diam/2)*(oldvalues[2]-initialY)), oldvalues[2]- Int(floor(diam/2)*(initialX-oldvalues[1])): oldvalues[2]+ Int(floor(diam/2)*(initialX-oldvalues[1])) ].=1
                    append!(diamArray,diam)
                  end
                end

                #println("if qq NOT " * string([initialX initialY]) * " " * string(firstneighborlist))
                if tim==1
                  firstneighborlist=vcat(firstneighborlist,[initialX initialY])
                end
                if tim>1 && [initialX,initialY] !=initialnod && abs(initialX-initialnod[1])<2 && abs(initialY-initialnod[2])<2
                  firstneighborlist=firstneighborlist[1:size(firstneighborlist,1)-1,:]
                  firstneighborlist=vcat(firstneighborlist,[initialX initialY])
                end
                break
            end
          end
          if tim==2000
           println("REACHED THE MAXIMUM tim!! "* string(ind) * " " * string(newNode) * "EXTREME CARE!!!" )
           initialX=initialnod[1]
           initialY=initialnod[2]
           traj=vcat(traj,[0 0])
          end
      end
      tmpList=[initialX initialY] .== outputNodes

      if tmpList[:,1]'*tmpList[:,2]==0 && [initialX initialY]!=initialnod
        outputNodes=vcat(outputNodes,[initialX initialY])
      end
      initialX, initialY=nod[newNode,1], nod[newNode,2]
      oldvalues=[initialX initialY]
      traj=vcat(traj, [initialX initialY])

      popfirst!(diamArray)
      if length(diamArray)==0
        length(diamMean)>2  ? append!(diamMean,diamMean[size(diamMean,1)]) : append!(diamMean,1)
      else
        append!(diamMean,mean(diamArray))
      end
      #println(size(diamMean))
      length(diamArray) < 100 ? append!(diamArray,zeros(100-length(diamArray))) : 1

      #append!(diamStat,diamArray)

    end
    return outputNodes[2:size(outputNodes,1),:], diamMean[2:size(diamMean,1)], traj
  end

  function NotInthelis(l, firstneighborlist)
    activate=true;
    for i=1:size(firstneighborlist)[1]
      if l ==firstneighborlist[i,:]'
        activate=false
        break
      end
    end
    return activate
  end

  function secondNeiIsNode(nod, x, y, oldx, oldy, initnode)
    r=0
    newx=0
    newy=0
    for j=-1:1, k=-1:1
      if [x+j,y+k]!=[oldx,oldy] && [x+j y+k]!= initnode
        arj= [x+j y+k].==nod
        r = arj[:,1]'*arj[:,2]

        if r==1
          newx=x+j
          newy=y+k
          break
        end
      end
    end

    if r==1
      return newx, newy, true
    else
      return x,y,false
    end
  end

  function NotANode(l,nod, firsttime, initNod)     #IF ACTIVATE IS TRUE, RETURN FALSE.
                                          # BASICALLY, IF l IS A NODE, THEN STOP THERE.
    activate=false;
    for i=1:size(nod)[1]
      if l==nod[i,:] && l!=initNod
        activate=true;
      end
    end

    if firsttime==1
      return true
    else
      if activate
        return false
      else
        return true
      end
    end
  end

  function NotANode2(nod, cNode, index)
    activate=false
    for i=1:size(cNode)[1]
      if nod[index]==cNode[i,:]
        activate=true;
      end
    end

      if activate
        return false
      else
        return true
      end
  end

  function getIndex(nod, x, y)
    aa=nod .==[x y]
    b=aa[:,1].*aa[:,2]
    ind =findall(x-> x==true, b)
    return ind[1]
  end

  function IterationOnAlgSearch(A, nod, numberOfSegPerNode, ASeg)
    #array1=zeros(size(A)[1],size(A)[2])
    #array1[nod[1,1],nod[1,2]]=1.0
    capturedNodes=nod[1,:]'
    adjMat=zeros(size(nod)[1],size(nod)[1])
    adjMatD=zeros(size(nod)[1],size(nod)[1])
    outputNodes, diamMean, traj=algSearch(A,nod,numberOfSegPerNode[1],1, ASeg)
    for j=1:size(outputNodes,1)
      vere, indie = calDist(nod, outputNodes, 1, j, traj)
      if vere
        outputNodes[j,:]=traj[indie,:]
        nod=vcat(nod,outputNodes[j,:]')
        append!(numberOfSegPerNode,2)
        adjMat=vcat(adjMat,zeros(1,size(adjMat,2)))
        adjMat=hcat(adjMat,zeros(size(adjMat,1),1))
        adjMatD=vcat(adjMatD,zeros(1,size(adjMatD,2)))
        adjMatD=hcat(adjMatD,zeros(size(adjMatD,1),1))
      end
    end
    #global trajTest=traj
    for i=1:size(outputNodes,1)
      x, y=outputNodes[i,1], outputNodes[i,2]
      #println([x,y])
      #array1[x,y]=1.0
      ind=getIndex(nod,x,y)
      #println(ind)
      adjMat[1,ind]=1.0
      adjMat[ind,1]=1.0
      adjMatD[1,ind]=diamMean[i]
      adjMatD[ind,1]=diamMean[i]
      #diamMean[i]==0 ? println("diam zero") : 1
    end

    nodesCp=copy(nod)
    for k=2:size(nodesCp)[1]
        outputNodes, diamMean, traj=algSearch(A,nod,numberOfSegPerNode[k],k,ASeg)
        for j=1:size(outputNodes,1)
          vere, indie = calDist(nod, outputNodes, k, j, traj)
          if vere
            outputNodes[j,:]=traj[indie,:]
            nod=vcat(nod,outputNodes[j,:]')
            append!(numberOfSegPerNode,2)
            adjMat=vcat(adjMat,zeros(1,size(adjMat,2)))
            adjMat=hcat(adjMat,zeros(size(adjMat,1),1))
            adjMatD=vcat(adjMatD,zeros(1,size(adjMatD,2)))
            adjMatD=hcat(adjMatD,zeros(size(adjMatD,1),1))
          end
        end
        #global trajTest=vcat(trajTest, traj)
        for i=1:size(outputNodes,1)
          x, y=outputNodes[i,1], outputNodes[i,2]
          #println([x,y])
          #array1[x,y]=1.0
          ind = getIndex(nod,x,y)
          #println(ind)
          adjMat[k,ind]=1
          adjMat[ind,k]=1
          adjMatD[k,ind]=diamMean[i]
          adjMatD[ind,k]=diamMean[i]
          #diamMean[i]==0 ? println("diam zero 2") : 1
        end
    end

    for k=size(nodesCp,1)+1:size(nod,1)
        outputNodes, diamMean, traj=algSearch(A,nod,numberOfSegPerNode[k],k,ASeg)
        # for j=1:size(outputNodes,1)
        #   vere, indie = calDist(nod, outputNodes, k, j, traj)
        #   if vere
        #     outputNodes[j,:]=traj[indie,:]
        #     nod=vcat(nod,outputNodes[j,:]')
        #     append!(numberOfSegPerNode,2)
        #     adjMat=vcat(adjMat,zeros(1,size(adjMat,2)))
        #     adjMat=hcat(adjMat,zeros(size(adjMat,1),1))
        #     adjMatD=vcat(adjMatD,zeros(1,size(adjMatD,2)))
        #     adjMatD=hcat(adjMatD,zeros(size(adjMatD,1),1))
        #   end
        # end
        #global trajTest=vcat(trajTest, traj)
        for i=1:size(outputNodes,1)
          x, y=outputNodes[i,1], outputNodes[i,2]
          #println([x,y])
          #array1[x,y]=1.0
          ind = getIndex(nod,x,y)
          #println(ind)
          adjMat[k,ind]=1
          adjMat[ind,k]=1
          adjMatD[k,ind]=diamMean[i]
          adjMatD[ind,k]=diamMean[i]
          #diamMean[i]==0 ? println("diam zero 3") : 1
        end
    end

    return nod, capturedNodes, adjMat, adjMatD
  end

  function measureDiam(aSeg, xold, yold, x, y)
    dr=[0 -1; 1 0]*[x-xold;y-yold]
    dmax=300
    d=0
    for i=1:dmax
      if aSeg[xold+dr[1]*i, yold+dr[2]*i]==0
        d=i
        break
      end
      if i==dmax
        println("something is wrong when calculating the diameter")
      end
    end
    for i=1:dmax
      if aSeg[xold-dr[1]*i, yold-dr[2]*i]==0
        d=d+i-1
        break
      end
      if i==dmax
        println("something is wrong when calculating the diameter")
      end
    end

    return d
  end

  function calDist(nod, outnod, i, j, traj)
    d= sqrt( (nod[i,1]-outnod[j,1])^2 + (nod[i,2]-outnod[j,2])^2 )
    d2=0
    c=0;
    index=0
    thrs=1.1
    for k=1:size(traj,1)
      if traj[k,:] == nod[i,:]
        c=0
        index=k
      end
      if traj[k,:] == outnod[j,:]
        d2=c
        index = index +Int(floor((k - index)/2))
        break
      end
      if sum(abs.(traj[k+1,:]-traj[k,:]))==1
        c=c+1
      elseif sum(abs.(traj[k+1,:]-traj[k,:]))>1
        c=c+sqrt(2)
      end
    end
    #println(string(i)*"   "* string(d) *"   " * string(d2) * "  " * string(nod[i,:]) * "   " * string(outnod[j,:]))
    if d2/d > thrs && d>10
      return true, index
    else
      return false, index
    end
  end

  function arrayCleanup(aMat, aMatD, nod, ep)
    n=[0,0]
    newaMat, newaMatD, newNod, ep = [], [],[],[]
    for i=1:size(aMat,1)
      if sum(aMat[i,:])==2
        count=1
        for j=1:size(aMat,2)
          if aMat[i,j]==1
            n[count]=j
            count=count+1
          end
        end

        if aMat[n[1],n[2]]==1 && sqrt( (nod[n[1],1]-nod[n[2],1])^2 + (nod[n[1],2]-nod[n[2],2])^2) < 10
          newaMat = vcat(aMat[1:i-1,:], aMat[i+1:size(aMat,1),:])
          newaMat = hcat(newaMat[:,1:i-1], newaMat[:,i+1:size(aMat,2)])
          newaMatD = vcat(aMatD[1:i-1,:], aMatD[i+1:size(aMat,1),:])
          newaMatD = hcat(newaMatD[:,1:i-1], newaMatD[:,i+1:size(aMat,2)])
          newNod=vcat(nod[1:i-1,:], nod[i+1:size(nod,1),:])
          for k=1:size(ep,1)
            ep[k]>i ? ep[k]=ep[k]-1 : 1
          end
          break
        end
      end
    end
    return newaMat, newaMatD, newNod, ep
  end
  ############################################DATA MANIPULATION
  function overImposedPic(a,nodes, ep)
    Aep=zeros(size(a,1),size(a,2))
    Anod=zeros(size(a,1),size(a,2))

    for i=1:size(ep,1)
      Aep[ep[i,1]-2:ep[i,1]+2, ep[i,2]-2:ep[i,2]+2] .=1
    end

    for i=1:size(nodes,1)
      Anod[nodes[i,1]-2:nodes[i,1]+2, nodes[i,2]-2:nodes[i,2]+2] .=1
    end

    return colorview(RGB, a, Anod, Aep)
  end

  function readAndDir(arg2, arg)
    list=readdir("./" * string(arg2) * "/")
    if (isdir(string(arg))) == false
       mkdir(string(arg))
     end
     return list, arg
  end

  function saveFig(pathh, fig)
    pathname= string(pathh)
    @show fig, pathname
    savefig(fig,pathname)
  end

  function saveData(pathh, dat)
    pathname=string(pathh)
    writedlm(pathname,dat)
  end
end
######################################################


function mymainFunc(thepath, readDir, writeDir)
  println("./"* string(readDir) * "/" * string(thepath) * "_Sket.tif")
    A=addZeros(init2("./"* string(readDir) * "/" * string(thepath) * "_Sket.tif"))
    ASeg=addZeros(init2("./"* string(readDir) * "/" * string(thepath) * ".gif"))
    #global diamMat=zeros(size(A,1),size(A,2))
    newA, nodes, arS, endpoints, numberOfSegPerNode=@time removeOverlapNodes(A )
    #colorview(RGB,A,arS,zeroarray)
    nodes, capNodes, adjMat, adjMatD=@time IterationOnAlgSearch(A,nodes,numberOfSegPerNode, ASeg)
    for i=1:size(adjMat,1)
      adjMat[i,i]=0
      adjMatD[i,i]=0
    end
    f(c)=round(c,digits=2)
    for i=1:size(adjMatD,1), j=1:size(adjMatD,2)
      if adjMat[i,j]>0 && adjMatD[i,j]==0
      end
    end

    positionEndpoints=[0]
    for i=1:size(nodes,1), j=1:size(endpoints,1)
      if endpoints[j,:]==nodes[i,:]
        append!(positionEndpoints,i-1)
      end
    end
    popfirst!(positionEndpoints)

    #adjMat,adjMatD,nodes,positionEndpoints=arrayCleanup(adjMat,adjMatD,nodes,positionEndpoints)

    loc2=[0 0];
    sstep=1;
    for i=1:sstep:size(A,1), j=1:sstep:size(A,1)
      if A[i,j]==1
        loc2=vcat(loc2,[j i])
      end
    end
    loc2=loc2[2:size(loc2)[1],:]
    # d=[scatter(x=nodes[:,1],y=nodes[:,2], marker_symbol="square", mode="markers")]
    # append!(d,[scatter(x=loc2[:,1],y=-loc2[:,2].+584, marker_symbol="square", mode="markers", marker_size=3,marker_color="red")])
    # for i=1:size(nodes)[1], j=1:size(nodes)[1]
    #   if adjMat[j,i]==1
    #     append!(d,[scatter(x=[nodes[j,2],nodes[i,2]],y=[584-nodes[j,1],584-nodes[i,1]], mode="line", marker_color="blue")])
    #   end
    # end
    # popfirst!(d)

    d = scatter(loc2[:,1],-loc2[:,2].+584, markershapes = :square, markerstrokewidth=0, frame=:box, size=(500,500), legend=false)
    for i=1:size(nodes)[1], j=1:size(nodes)[1]
      if adjMat[j,i]==1
        # append!(d,[scatter(x=[nodes[j,2],nodes[i,2]],y=[584-nodes[j,1],584-nodes[i,1]], mode="line", marker_color="blue")])
        d = plot!([nodes[j,2],nodes[i,2]],[584-nodes[j,1],584-nodes[i,1]],
              markershapes = :circle, lw=2, ms=5, legend=:none, c=:black, markerstrokewidth=0)
      end
    end
    fig=plot(d)
    saveFig(writeDir * "/" * thepath[1:10]*"_network.png", fig)
    save(writeDir * "/" * thepath[1:10]*"_Pic.png",overImposedPic(A,nodes, endpoints))
    # writedlm(thepath[1:10]*"_adj.csv",Array{Int,2}(adjMat))
    # writedlm(thepath[1:10]*"_nodes.csv",hcat(nodes,[10.0 for i=1:size(nodes,1)]))
    # writedlm(thepath[1:10]*"_endpoints.csv",Array{Int,1}(positionEndpoints))
    saveData(writeDir * "/" * thepath[1:10]*"_adj.csv",Array{Int,2}(adjMat))
    saveData(writeDir * "/" * thepath[1:10]*"_adjD.csv",Array{Float16,2}(f.(adjMatD)))
    saveData(writeDir * "/" * thepath[1:10]*"_nodes.csv",hcat(nodes,[10.0 for i=1:size(nodes,1)]))
    saveData(writeDir * "/" * thepath[1:10]*"_endpoints.csv",Array{Int,1}(positionEndpoints))
    return d, adjMat, positionEndpoints, nodes, adjMatD, endpoints
end


function theWholeProcess(readDir,writeDir)
  list, outDir=readAndDir( readDir, writeDir)
  list=setdiff([list[i][1:10] for i=1:size(list,1)])

  for i in list
    d, adjMat, endpoints, nodes, adjMatD, ep=mymainFunc(i, readDir,writeDir)

  end
  return 1

end

plot([0,1],[2,6],
      markershapes = :circle, lw=2, ms=8, legend=:none, c=:red, markerstrokewidth=0)
pwd()
###########################################################
######################################################
#######################################################
#######################################################
######################################################
#######################################################
thepath="./sket/01_manual1_Sket.tif"
thepath="01_manual1"
#diamMat=zeros(10,10)
#diamStat=[0]
#trajTest=[0 0]
cd("/Users/javier/Documents/Retinopathy/DRIVE/MySeg")
# cd("../Retinopathy/DRIVE/MySeg/") #Change path to where
                      #this script, the sket dir and the network dir is
@time d, adjMat, endpoints, nodes, adjMatD, ep=mymainFunc(thepath, "sket", "networks")
plot(d)
@time d=mymainFunc(thepath, "sket", "networks")
# This function will go over all the skeletonized segmented .tif images
# and compute the Adj Matrix, end points
theWholeProcess("sket", "networks")

#=INFO
d is a plot of the computed network overimposed on the original skeletonized image

adjMat = adjacency matrix, i.e. adjMat[i,j] = 1 if node i and node j are connected

endpoints is a list of the positions in nodes that are connected to only 1 node

nodes = positions of the nodes

adjMatD = adjacency matrix where adjMat[i,j] = diameter if node i and node
j are connected, otherwise gives 0

ep is an array of the positions of the endpoints. endpoints[i] is located at
(ep[i,1],ep[i,1])
=#




################################################
# Aseg=zeros(size(addZeros(init2("01_manual1_SketA.tif")),1),size(addZeros(init2("01_manual1_SketA.tif")),2))
# A=addZeros(init2("01_manual1_SketA.tif"))
#
# colorview(RGB,addZeros(init2("01_manual1.gif")),Apath,Aseg)
# colorview(RGB,addZeros(init2("01_manual1_SketA.tif")),zeroarray,diamMat)
# colorview(RGB,addZeros(init2("01_manual1.gif")),Aseg,diamMat)
# colorview(RGB,Aseg,Apath,diamMat)
# colorview(RGB,addZeros(init2("01_manual1.gif")),Apath,diamMat)
# colorview(Gray,Apath)
#
# for i=1:size(ep,1)
#   # println(nodes[i, :])
#   Aseg[ep[i,1]-2:ep[i,1]+2, ep[i,2]-2:ep[i,2]+2] .=1
# end
#
# Apath=zeros(size(addZeros(init2("01_manual1_SketA.tif")),1),size(addZeros(init2("01_manual1_SketA.tif")),2))
# for i=1:size(nodes,1)
#   # println(nodes[i, :])
#   # if trajTest[i,:]!=[0, 0]
#   #   Apath[trajTest[i,1]:trajTest[i,1], trajTest[i,2]:trajTest[i,2]] .=1
#   # end
#   Apath[nodes[i,1]-2:nodes[i,1]+2, nodes[i,2]-2:nodes[i,2]+2] .=1
# end
