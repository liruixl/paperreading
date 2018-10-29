#ifndef _DPU_ENTITY
#define _DPU_ENTITY

module GlobalEntity {

	//ͼ���ڴ����
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
		string		 iNMatNO;			//��ڲ��Ϻ�		C20
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
		string		iNMatNO;			//��ڲ��Ϻ�		C20
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
		int			grayScaleX;		//ˮƽ�ü���ֵ
		int			grayScaleY;		//��ֱ�ü���ֵ
		int			dropWidth;		//���������С��dropWidth���������
		int			dropHeight;		//�������߶�С��dropHeight���������
		int			gaussSize;		//��˹�˴�С
		int			sxThreshold;	//sobelX��ֵ
		int			syHighThreshold;//sobely����ֵ
		int			syLowThreshold;	//sobely����ֵ
		int			binaryThreshold;//��ֵ����ֵ
		int			clusterX;		//X������ಽ��
		int			clusterY;		//Y������ಽ��
		int			pixelsMin;		//����������
	};
	
	struct PlcInfo{
		int 		speed;
		int			position;	 //λ��ֵ����ʼ����ʱ�̣��ӹ���ź�̽�⵽�иֹܾ���ʱ
		//float       temperature;
		//renzeyu20130213
		int POS1SetRem;			//1#��������λ���趨
		int POS2SetRem;			//2#��������λ���趨
		int POS3SetRem;			//3#��������λ���趨
		int POS4SetRem;			//4#��������λ���趨
		int POS5SetRem;			//5#��������λ���趨
		int POS6SetRem;			//6#��������λ���趨
		int POS7SetRem;			//������С��λ���趨
		int POS8SetRem;			//������С��λ���趨
		int HeartBitR;			//����λ
		int LocalR;				//���Բ���ģʽ
		int RemotR;				//Զ�̲���ģʽ
		int POS1RealR;			//1#��������ʵ��λ��
		int POS2RealR;			//2#��������ʵ��λ��
		int POS3RealR;			//3#��������ʵ��λ��
		int POS4RealR;			//4#��������ʵ��λ��
		int POS5RealR;			//5#��������ʵ��λ��
		int POS6RealR;			//6#��������ʵ��λ��
		int POS7RealR;			//������С��ʵ��λ��
		int POS8RealR;			//������С��ʵ��λ��
		int State1R;			//1#�������״̬����
		int State2R;			//2#�������״̬����
		int State3R;			//3#�������״̬����
		int State4R;			//4#�������״̬����
		int State5R;			//5#�������״̬����
		int State6R;			//6#�������״̬����
		int State7R;			//��������״̬����
		int State8R;			//��������״̬����
		int Temp1R;				//1#������¶�
		int Temp2R;				//2#������¶�
		int Temp3R;				//3#������¶�
		int Temp4R;				//4#������¶�
		int Temp5R;				//5#������¶�
		int Temp6R;				//6#������¶�
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
		string INMatNO;			//��ڲ��Ϻ�		C20
		int    DefectNum;		//ȱ����
		int    CircleDefectNum; //������ȱ����
		int    SpecialDefectNum;//С���ȱ����
		int    MaxCounter;
		long   StartTime;		//��ʼʱ��
		long   EndTime;			//����ʱ��
		long   timeInMs;		//����ʱ��
		//renzeyu20170113
		string RollingLot;		//����		C6
		string RollingSpecification;	//���		C12
		string SteelGrade;		//����		C4
		string HeatNumber;		//¯��		C8
		string RollingCount;	//����		C4
		string SerialNumber;	//���		C4
		string TubeDiameter;	//�����⾶	N(3,2)	��λ��MM
		string TubeThickness;	//���Ʊں�	N(2,2)	��λ��MM
		string GrooveCode;		//�ӿ���		C5
		string SRWLength;		//���۳��ȣ��ż�����		N(3,2) ��λ:M
		//20170215 shangyufei ��������1����2
		string Spare1; 			//����1
		string Spare2;			//����2
	};
	
	sequence<GlobalEntity::MatertialInfo> MatertialInfoList;

	struct MatertialInfoValue{
		long   id;
		string INMatNO;			//��ڲ��Ϻ�		C20
		long   timeInMs;		//����ʱ��
		//renzeyu20170113
		string RollingLot;		//����		C6
		string RollingSpecification;	//���		C12
		string SteelGrade;		//����		C4
		string HeatNumber;		//¯��		C8
		string RollingCount;	//����		C4
		string SerialNumber;	//���		C4
		double TubeDiameter;	//�����⾶	N(3,2)	��λ��MM
		double TubeThickness;	//���Ʊں�	N(2,2)	��λ��MM
		string GrooveCode;		//�ӿ���		C5
		string SRWLength;		//���۳��ȣ��ż�����		N(3,2) ��λ:M
		//20170215 shangyufei ��������1����2
		string Spare1; 			//����1
		string Spare2;			//����2
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