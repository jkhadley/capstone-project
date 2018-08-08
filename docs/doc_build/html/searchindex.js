Search.setIndex({docnames:["callbacks","generators","index","metrics","misc","modelInferencer","modelTrainer","modules","plottingFunctions","regressionPreprocessData","segmentationPreprocessData"],envversion:53,filenames:["callbacks.rst","generators.rst","index.rst","metrics.rst","misc.rst","modelInferencer.rst","modelTrainer.rst","modules.rst","plottingFunctions.rst","regressionPreprocessData.rst","segmentationPreprocessData.rst"],objects:{"":{callbacks:[0,0,0,"-"],generators:[1,0,0,"-"],metrics:[3,0,0,"-"],misc:[4,0,0,"-"],modelInferencer:[5,0,0,"-"],modelTrainer:[6,0,0,"-"],plottingFunctions:[8,0,0,"-"],regressionPreprocessData:[9,0,0,"-"],segmentationPreprocessData:[10,0,0,"-"]},"callbacks.BatchLogger":{on_batch_end:[0,2,1,""],on_epoch_end:[0,2,1,""]},"modelInferencer.ModelInferencer":{batchPredict:[5,2,1,""],buildBatch:[5,2,1,""],getConfusionMatrix:[5,2,1,""],getRegressionAccuracy:[5,2,1,""],getSegmentationAccuracy:[5,2,1,""],imageSetGenerator:[5,2,1,""],predict:[5,2,1,""],regressionBatchError:[5,2,1,""],regressionPredict:[5,2,1,""],segmentationBatchError:[5,2,1,""],segmentationPredict:[5,2,1,""],setBatchSize:[5,2,1,""],setClassMap:[5,2,1,""]},"modelTrainer.ModelTrainer":{batch_size:[6,4,1,""],buildCallbacks:[6,2,1,""],changeBatchLogInterval:[6,2,1,""],changeBatchSize:[6,2,1,""],changeConvolutionalDepth:[6,2,1,""],changeDataPath:[6,2,1,""],changeDropout:[6,2,1,""],changeEpochs:[6,2,1,""],changeInputShape:[6,2,1,""],changeLossFunction:[6,2,1,""],changeMetrics:[6,2,1,""],changeModelSavePath:[6,2,1,""],changeResultsSavePath:[6,2,1,""],continueTraining:[6,2,1,""],conv_depth:[6,4,1,""],createModel:[6,2,1,""],evaluate:[6,2,1,""],printParameters:[6,2,1,""],setClassMap:[6,2,1,""],setClassName:[6,2,1,""],setGenerators:[6,2,1,""],setOldModel:[6,2,1,""],setOptimizerParams:[6,2,1,""],setRegression:[6,2,1,""],setSaveName:[6,2,1,""],setSegmentation:[6,2,1,""],setWeightInitializer:[6,2,1,""],singlePrediction:[6,2,1,""],train:[6,2,1,""]},callbacks:{BatchLogger:[0,1,1,""]},generators:{batchGenerator:[1,3,1,""],getBatchGenerators:[1,3,1,""],multiClassGenerator:[1,3,1,""],singleClassGenerator:[1,3,1,""]},metrics:{RMSE:[3,3,1,""],f1Score:[3,3,1,""],precision:[3,3,1,""],recall:[3,3,1,""]},misc:{combineFiles:[4,3,1,""],countFiles:[4,3,1,""],dirSize:[4,3,1,""]},modelInferencer:{ModelInferencer:[5,1,1,""]},modelTrainer:{ModelTrainer:[6,1,1,""],baseUNet:[6,3,1,""],getPropOfGround:[6,3,1,""]},plottingFunctions:{getTrainPredictions:[8,3,1,""],plotBatchMetrics:[8,3,1,""],plotEpochMetrics:[8,3,1,""],plotPredictions:[8,3,1,""],setGenerator:[8,3,1,""]},regressionPreprocessData:{calcPropOfGround:[9,3,1,""],consolidateAllFiles:[9,3,1,""],consolidateFiles:[9,3,1,""],getAllErrors:[9,3,1,""],getErrors:[9,3,1,""],savePropOfGround:[9,3,1,""]},segmentationPreprocessData:{makeSplitDirs:[10,3,1,""],preprocessData:[10,3,1,""],randomSplit:[10,3,1,""],renameLabels:[10,3,1,""],splitImageMP:[10,3,1,""],splitImagesIntoDirectories:[10,3,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute"},terms:{"boolean":1,"break":5,"class":[0,1,2,7,8,10],"default":[1,3,6],"final":2,"float":[3,6,10],"function":[2,6,7,10],"int":[0,1,4,5,6,8,10],"new":6,"return":[1,3,4,5,6,8,10],"switch":6,"true":3,"try":2,Adding:2,The:[1,2,3,5,6,10],These:[2,10],Useful:4,Uses:6,accept:6,accord:6,accuraci:2,actual:8,add:10,added:2,after:10,again:10,all:6,alreadi:10,also:2,ani:[5,8],append:4,appropri:[1,5,6,10],approxim:10,argument:10,arrai:[1,3,5,6,8],assum:10,avoid:2,awar:10,back:5,base:[0,1,5,6,10],baseunet:6,basic:6,batch:[0,1,5,6],batch_f:0,batch_siz:[1,5,6],batchgener:1,batchinterv:0,batchlogg:0,batchpredict:5,batchsiz:8,been:10,befor:[3,6],being:1,belong:5,between:0,build:[5,6],buildbatch:5,buildcallback:6,built:2,calcpropofground:9,calcul:3,call:10,callback:[2,6,7],can:2,categor:[1,6],categori:10,ce_nadir_:10,chang:6,changebatchloginterv:6,changebatchs:6,changeconvolutionaldepth:6,changedatapath:6,changedropout:6,changeepoch:6,changeinputshap:6,changelossfunct:6,changemetr:6,changemodelsavepath:6,changeresultssavepath:6,checkpoint:6,class1:[2,10],class2:10,classifi:1,classmap:[1,5,6,8],classnam:[1,10],combin:4,combinefil:4,come:5,compil:6,consolidateallfil:9,consolidatefil:9,contain:[1,4,6,8],continuetrain:6,conv_depth:6,convolut:6,copi:3,correct:[5,6],correctclass:8,correspond:6,could:10,count:4,countfil:4,creat:[1,6,10],createmodel:6,crop:8,cross:6,csv:[6,8],custom:[2,7],cycl:1,data:[2,5,6,7],data_s:1,data_shap:1,datadir:[2,10],datapath:[5,6,10],deal:10,decai:6,defin:[8,10],defualt:6,depth:6,determin:[2,5],dictionari:[0,1,5,6,8,10],differ:8,dir:4,directli:10,directori:[1,2,4,5,6,10],dirsiz:4,divid:2,doesn:10,done:[2,10],drop:6,dropout:6,each:[1,2,5,6,8,10],effect:10,either:1,end:0,entropi:6,epoch:[0,6],epoch_f:0,error:3,evalu:[0,6],evaulu:6,exampl:[2,7],extra:2,f1score:3,fals:1,feed:[1,10],fig_height:8,figur:8,file:[0,2,4,6,10],find:[4,6],finish:0,first:[4,6],fit:5,folder:[1,8],frequent:0,from:[1,3,4,5,6,8,10],full:10,further:[2,6],gener:[2,5,6,7],get:[1,5,6,10],getallerror:9,getbatchgener:1,getconfusionmatrix:5,geterror:9,getpropofground:6,getregressionaccuraci:5,getsegmentationaccuraci:5,gettrainpredict:8,give:[6,10],given:[2,3],goe:10,ground:[5,6],groundcover2016:10,had:10,has:[6,10],height:8,here:10,how:10,ignor:[6,10],ignoredir:10,ignoredirectori:10,imag:[1,2,4,5,6,8,10],imagesetgener:5,imageshap:10,img:[5,6,8],improv:2,includ:4,index:[2,10],indic:[1,10],indirectli:2,infer:5,inform:6,init_w:6,initi:[0,2,6],input:[3,4,6,8],input_shap:6,intact:10,intend:10,interest:10,interv:6,intial:6,iter:[1,6],its:5,jpg:10,just:0,keep:2,kera:[0,2,3,5,6,8],kwarg:1,label:[1,2,5,8,10],label_s:1,label_shap:1,labeloutput:1,labelshap:1,larg:10,layer:[2,6],learn:6,leav:10,length:[4,10],like:[2,10],list:[0,4,6,10],load:[2,5,6],locat:[1,4,6,10],log:[0,6],logger:[0,6],look:6,loss:6,made:8,maiz:10,maizevarieti:10,make:[5,6,8,10],makesplitdir:10,mani:[6,10],map:[1,5,6],match:10,mean:3,measur:0,method:6,metric:[0,2,6,7],middl:6,million:2,misc:[2,7],model:[0,1,2,5,6,8],modelinferenc:[2,7],modelpath:6,modeltrain:[2,7],modul:2,momentum:6,more:[0,6],mse:6,multi:1,multiclassgener:1,multipl:[4,5],mungbean:10,n_class:6,name:[1,2,4,6,10],need:10,net:6,network:2,neural:2,node:6,none:1,num_of_img:8,number:[0,2,4,5,6,8,10],object:[5,6],old:6,on_batch_end:0,on_epoch_end:0,onli:10,open:2,optim:6,order:10,origin:[8,10],other:10,out:1,output:[1,2,3,5,6,8],overal:8,page:2,pair:[1,10],param:[5,8,9,10],paramet:[0,1,3,4,5,6,8,10],pass:1,path:[1,4,5,6,8,10],pattern:10,per:5,perform:0,piec:[5,10],pixel:2,place:[2,10],plot:[2,7],plotbatchmetr:8,plotepochmetr:8,plotpredict:8,plottingfunct:8,precis:3,pred:[5,8],predict:[2,3,5,6,8],preprocess:[2,7],preprocessdata:10,primari:10,print:6,printparamet:6,problem:2,produc:1,proport:[2,6,10],proprocess:10,proptrain:10,propvalid:10,provid:8,put:[0,10],question:4,quickli:2,ram:2,random:8,randomli:6,randomsplit:10,rate:6,recal:3,record:0,regress:[1,2,5,6],regressionbatcherror:5,regressionpredict:5,regressionpreprocessdata:[],reiter:1,remov:[3,10],renam:10,renamelabel:10,replac:10,result:[0,4,6],resultspath:6,right:10,rmse:3,root:3,rsme:3,run:[1,10],same:6,save:[4,6],savenam:6,savepropofground:9,score:3,search:2,second:4,see:6,segment:[2,5,6],segmentationbatcherror:5,segmentationpredict:5,segmentationpreprocessdata:10,semant:2,separ:10,set:[1,5,6,10],setbatchs:5,setclassmap:[5,6],setclassnam:6,setgener:[6,8],setoldmodel:6,setoptimizerparam:6,setregress:6,setsavenam:6,setsegment:6,setweightiniti:6,sgd:6,shape:[1,6,8,10],should:[6,8,10],show:[],singl:[1,6],singleclassgener:1,singlepredict:6,size:[1,2,4,5,6,8,10],small:2,smaller:10,somewhat:2,sourc:[0,1,3,4,5,6,8,9,10],specifi:[1,4,6,8,10],split:10,splitimagemp:10,splitimagesintodirectori:10,squar:3,src:9,start:[1,10],still:2,stitch:[5,8],store:6,string:[0,1,4,5,6,8,10],structur:[2,10],sub:10,subdir1:[2,10],subdir2:[2,10],subdirectori:[2,10],subdirs:10,subimgs:8,sum:[2,5],take:[8,10],test:[4,10],testind:10,thei:[2,3,10],them:[2,6,8,10],thi:[2,3,10],through:[1,6,10],tif:10,time:6,togeth:[5,8],too:10,train:[1,2,4,6,10],train_path:8,traingener:1,trainind:10,trainprop:10,trainproport:10,transform:8,truth:5,truthclass:5,tupl:[1,6,10],turn:2,twice:10,type:[1,3,4,5,6,8],uniqu:1,until:1,updat:6,use:[6,8,10],used:[0,6,8,10],uses:2,using:[2,6,8],valid:[1,4,6,10],validata:4,validateproport:10,validationgener:1,valind:10,valprop:10,valu:8,variabl:[6,10],variou:5,via:[1,5],weight:6,what:10,wheat:10,when:[1,10],where:[5,6,10],which:[0,4,5,6,10],whichclass:10,whichdir:[6,10],whichset:5,width:6,window:8,work:2,write_dest:9,written:6,y_pred:3,y_true:3,yield:[1,5]},titles:["Custom Callbacks","Data Generators","Welcome to capstone-project\u2019s documentation!","Metrics","Misc. Functions","ModelInferencer Class","ModelTrainer Class","&lt;no title&gt;","Plotting Functions","regressionPreprocessData module","Preprocess Data"],titleterms:{"class":[5,6],"function":[4,8],callback:0,capston:2,custom:0,data:[1,10],document:2,exampl:10,gener:1,indic:2,metric:3,misc:4,modelinferenc:5,modeltrain:6,modul:9,plot:8,plottingfunct:[],preprocess:10,project:2,python:[],regressionpreprocessdata:9,segmentationpreprocessdata:[],tabl:2,test:[],welcom:2}})