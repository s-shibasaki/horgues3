using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace JVDataCollector
{
    public partial class JVLinkForm : Form
    {
        private string[] commandLineArgs;

        public JVLinkForm(string[] args)
        {
            InitializeComponent();
            commandLineArgs = args;
        }

        private void JVLinkForm_Load(object sender, EventArgs e)
        {
            try
            {
                if (commandLineArgs.Length > 0)
                {
                    string command = commandLineArgs[0].ToLower();
                    switch (command)
                    {
                        case "setup":
                            ExecuteSetup();
                            break;
                        case "update":
                            ExecuteUpdate();
                            break;
                        default:
                            throw new ArgumentException($"Unknown command: {command}");
                    }
                }
                else
                {
                    throw new ArgumentException("No command line arguments provided. Usage: program.exe [setup|update]");
                }
            }
            finally
            {
                Application.Exit();
            }
        }

        private void ProcessJVData(string dataSpec, string fromTime, int option)
        {
            int result;
            int filesToRead = 0, filesToDownload = 0;
            string lastFileTimestamp;

            result = axJVLink1.JVOpen(dataSpec, fromTime, option, ref filesToRead, ref filesToDownload, out lastFileTimestamp);
            if (result != 0)
            {
                throw new InvalidOperationException($"JVOpen ({dataSpec}) failed with error code: {result}");
            }

            // JVOpen実行後の情報をコンソールに出力
            Console.WriteLine($"JVOpen executed successfully:");
            Console.WriteLine($"  DataSpec: {dataSpec}");
            Console.WriteLine($"  FromTime: {fromTime}");
            Console.WriteLine($"  Option: {option}");
            Console.WriteLine($"  FilesToRead: {filesToRead}");
            Console.WriteLine($"  FilesToDownload: {filesToDownload}");
            Console.WriteLine($"  LastFileTimestamp: {lastFileTimestamp}");
            Console.WriteLine();

            // TODO: JVStatus, JVRead を行う 

            // TODO: lastFileTimestamp をデータベースに保存する

            axJVLink1.JVClose();
        }

        private void ExecuteSetup()
        {
            int result;
            StringBuilder sb = new StringBuilder();
            string dataSpec, fromTime;

            result = axJVLink1.JVInit("SA000000/SD000004");
            if (result != 0)
            {
                throw new InvalidOperationException($"JVInit failed with error code: {result}");
            }

            // 読み出し終了ポイントを指定できないデータ種別
            sb.Clear();
            sb.Append("TOKU");
            sb.Append("DIFN");
            sb.Append("HOSN");
            sb.Append("HOYU");
            sb.Append("COMM");
            dataSpec = sb.ToString();
            fromTime = "19860101000000";
            ProcessJVData(dataSpec, fromTime, 4);

            // 読み出し終了ポイントを指定できるデータ種別は1年分ずつ取得
            sb.Clear();
            sb.Append("RACE");
            sb.Append("SLOP");
            sb.Append("WOOD");
            sb.Append("YSCH");
            sb.Append("MING");
            sb.Append("BLDN");
            sb.Append("SNPN");
            dataSpec = sb.ToString();

            // 1986年から去年まで1年ずつ取得
            int currentYear = DateTime.Now.Year;
            for (int year = 1986; year < currentYear; year++)
            {
                fromTime = $"{year}0101000000-{year + 1}0101000000";
                ProcessJVData(dataSpec, fromTime, 4);
            }

            // 今年の分は終了時刻を指定しない
            fromTime = $"{currentYear}0101000000";
            ProcessJVData(dataSpec, fromTime, 4);
        }

        private void ExecuteUpdate()
        {
            throw new NotImplementedException("Update functionality is not yet implemented");
        }
    }
}
