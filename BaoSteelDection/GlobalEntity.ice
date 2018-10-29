#ifndef _DPU_ENTITY
#define _DPU_ENTITY

module GlobalEntity {

	//图像内存对象
	sequence<byte> ImgBuffer;

	enum ResultStatus{
		FileSaveUpload,
		FileSaveNoUpload,
		FileDrop
	};

	struct DPUImage{
		int			 height;
		int			 width;
		int			 resolution;
		ImgBuffer	 buffer;
		string		 label;
		int			 counter;
		int			 channel;
		bool         location;
		bool		 location2;
		bool         location3;
		bool		 location4;
		bool         location5;
		bool		 location6;
		long		 timeInMs;
		string		 dpuID;
		long		 matertialId;
		string		 iNMatNO;			//入口材料号		C20
		string		 sStr;
		double		 sDou;
		string		 spare;
	};

	enum DefectClass{ 
		DefectNone,
		DefectLine, 
		DefectSquare	
	};

	enum HardDiskStorageStatus{
		StorageLow,
		StorageNormal,
		StorageHigh,
		StorageWarn,
		StorageDanger
	};

	struct DPUStatusEntity{
		HardDiskStorageStatus	status;
		float					memoryRate;
		float					cpuRate;
		//float					diskRate;
		int						unfinishedTask;
		int						waitedTask;
		int						unsavedTask;
		int						cameraStatus;
	};

	struct DefectObject{
		long		id;
		int			height;
		int			width;
		int			shape;
		int			subshape;
		int			level;
		int			pixelNum;
		double		area;
		int			posX;
		int			posY;
		int  		counter;
		string		dpuID;
		bool		isDisplay;
		string		imageLabel;	
		long		matertialId;	
		string		iNMatNO;			//入口材料号		C20
	};
	
	sequence<GlobalEntity::DefectObject> DefectObjectList;

	struct DPUParameters{
		//int			gaussSize;
		//int			pixelsMin;
		//int			binaryThreshold;
		//float		resizeX;
		//float		resizeY;
		//int			grayScale;
		//float		scale;
		//
		int			grayScaleX;		//水平裁剪阈值
		int			grayScaleY;		//垂直裁剪阈值
		int			dropWidth;		//若钢条宽度小于dropWidth则跳过检测
		int			dropHeight;		//若钢条高度小于dropHeight则跳过检测
		int			gaussSize;		//高斯核大小
		int			sxThreshold;	//sobelX阈值
		int			syHighThreshold;//sobely高阈值
		int			syLowThreshold;	//sobely低阈值
		int			binaryThreshold;//二值化阈值
		int			clusterX;		//X方向聚类步长
		int			clusterY;		//Y方向聚类步长
		int			pixelsMin;		//最少像素数
	};
	
	struct PlcInfo{
		int 		speed;
		int			position;	 //位置值的起始计算时刻，从光电信号探测到有钢管经过时
		//float       temperature;
		//renzeyu20130213
		int POS1SetRem;			//1#检测箱相机位置设定
		int POS2SetRem;			//2#检测箱相机位置设定
		int POS3SetRem;			//3#检测箱相机位置设定
		int POS4SetRem;			//4#检测箱相机位置设定
		int POS5SetRem;			//5#检测箱相机位置设定
		int POS6SetRem;			//6#检测箱相机位置设定
		int POS7SetRem;			//操作侧小车位置设定
		int POS8SetRem;			//传动侧小车位置设定
		int HeartBitR;			//心跳位
		int LocalR;				//机旁操作模式
		int RemotR;				//远程操作模式
		int POS1RealR;			//1#检测箱相机实际位置
		int POS2RealR;			//2#检测箱相机实际位置
		int POS3RealR;			//3#检测箱相机实际位置
		int POS4RealR;			//4#检测箱相机实际位置
		int POS5RealR;			//5#检测箱相机实际位置
		int POS6RealR;			//6#检测箱相机实际位置
		int POS7RealR;			//操作侧小车实际位置
		int POS8RealR;			//传动侧小车实际位置
		int State1R;			//1#检测箱电机状态代码
		int State2R;			//2#检测箱电机状态代码
		int State3R;			//3#检测箱电机状态代码
		int State4R;			//4#检测箱电机状态代码
		int State5R;			//5#检测箱电机状态代码
		int State6R;			//6#检测箱电机状态代码
		int State7R;			//操作侧电机状态代码
		int State8R;			//传动侧电机状态代码
		int Temp1R;				//1#检测箱温度
		int Temp2R;				//2#检测箱温度
		int Temp3R;				//3#检测箱温度
		int Temp4R;				//4#检测箱温度
		int Temp5R;				//5#检测箱温度
		int Temp6R;				//6#检测箱温度
	};

	struct SteelRollingInfo{
		long		id;
		string		label;
		double		length;
		double		diameter;
		long		times;
		int			unitNo;
	};
	
	sequence<GlobalEntity::SteelRollingInfo> SteelRollingList;

	sequence<string> FileList;

	
	struct MatertialInfo{
		long   id;
		string INMatNO;			//入口材料号		C20
		int    DefectNum;		//缺陷数
		int    CircleDefectNum; //周期性缺陷数
		int    SpecialDefectNum;//小类别缺陷数
		int    MaxCounter;
		long   StartTime;		//开始时间
		long   EndTime;			//结束时间
		long   timeInMs;		//插入时间
		//renzeyu20170113
		string RollingLot;		//轧批		C6
		string RollingSpecification;	//规格		C12
		string SteelGrade;		//材质		C4
		string HeatNumber;		//炉号		C8
		string RollingCount;	//根数		C4
		string SerialNumber;	//序号		C4
		string TubeDiameter;	//轧制外径	N(3,2)	单位：MM
		string TubeThickness;	//轧制壁厚	N(2,2)	单位：MM
		string GrooveCode;		//子孔型		C5
		string SRWLength;		//理论长度（张减机后）		N(3,2) 单位:M
		//20170215 shangyufei 新增备用1备用2
		string Spare1; 			//备用1
		string Spare2;			//备用2
	};
	
	sequence<GlobalEntity::MatertialInfo> MatertialInfoList;

	struct MatertialInfoValue{
		long   id;
		string INMatNO;			//入口材料号		C20
		long   timeInMs;		//插入时间
		//renzeyu20170113
		string RollingLot;		//轧批		C6
		string RollingSpecification;	//规格		C12
		string SteelGrade;		//材质		C4
		string HeatNumber;		//炉号		C8
		string RollingCount;	//根数		C4
		string SerialNumber;	//序号		C4
		double TubeDiameter;	//轧制外径	N(3,2)	单位：MM
		double TubeThickness;	//轧制壁厚	N(2,2)	单位：MM
		string GrooveCode;		//子孔型		C5
		string SRWLength;		//理论长度（张减机后）		N(3,2) 单位:M
		//20170215 shangyufei 新增备用1备用2
		string Spare1; 			//备用1
		string Spare2;			//备用2
	};
	
	sequence<GlobalEntity::MatertialInfoValue> MatertialInfoValueList;

	struct BackupInfo
	{
		string tableName;
		long defectNum;
		long defectIdStart;
		long defectIdEnd;
		long matertialIdStart;
		long matertialIdEnd;
		short isBackup;
		short isExist;
	};

	sequence<GlobalEntity::BackupInfo> BackupList;

	struct DefectShapeInfo
	{
		int id;
		string defectName;
		short isDelete;
	};

	sequence<GlobalEntity::DefectShapeInfo> DefectClassList;
	
	sequence<long> BigIntIdList;
	
	sequence<int> IntIdList;


};

#endif