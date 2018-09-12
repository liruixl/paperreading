进度条窗体

```c#
using System;
using System.Windows.Forms;

namespace BSClassifier
{
    public partial class progressbar : Form
    {
        //#qu 修改单例

        public Form OwnerForm; //父窗体

        public progressbar()
        {
            InitializeComponent();
        }

        public progressbar(int _Minimum, int _Maximum, Form _OwnerForm)//带参数，表示进度条的范围的最小值和最大值
        {
            InitializeComponent();
            this.StartPosition = FormStartPosition.CenterParent;
            progressBar1.Maximum = _Maximum;//设置范围最大值
            progressBar1.Value = progressBar1.Minimum = _Minimum;//设置范围最小值
            this.OwnerForm = _OwnerForm;
        }
        public void setPos(int value)//设置进度条当前进度值
        {
            if (value <= progressBar1.Maximum)//如果值有效
            {
                progressBar1.Value = value;//设置进度值
                if(progressBar1.Maximum!=0)
                    label1.Text = (value * 100 / progressBar1.Maximum).ToString() + "%";//显示百分比
            }
            Application.DoEvents();//重点，必须加上，否则父子窗体都假死
        }
        private void progressbar_Load(object sender, EventArgs e)
        {
            this.OwnerForm.Enabled = false;//设置父窗体不可用
        }
        private void progressbar_FormClosed(object sender, FormClosedEventArgs e)
        {
            this.OwnerForm.Enabled = true;//回复父窗体为可用
        }
    }
}

```

